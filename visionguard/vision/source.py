"""Frame sources: deterministic synthetic scenes and real video/RTSP inputs.

The synthetic renderer exists so the system can be demonstrated, benchmarked
and tested without a camera.  It draws actual pixels (people, vehicles, static
objects) over a noisy background, so every downstream stage - background
subtraction, tracking, features, autoencoder - is exercised for real.  Ground
truth anomaly labels are exposed only for benchmarking; they never influence
detection.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from visionguard.config import CameraConfig
from visionguard.core.types import BadgeSignal

ANOMALY_KINDS = (
    "running",
    "crowd",
    "loitering",
    "abandoned",
    "tailgate",
    "vehicle",
    "after_hours",
)

_WALK_BAND = (0.46, 0.68)
_SCENE_SPAWN = {
    "lobby": (1.8, 4.2),
    "parking": (2.6, 5.5),
    "server": (7.0, 14.0),
    "office": (3.5, 7.5),
    "loading": (3.0, 6.0),
    "stairwell": (5.0, 10.0),
}


class FrameSource(Protocol):
    width: int
    height: int
    fps: float

    def open(self) -> None: ...
    def read(self) -> np.ndarray | None: ...
    def close(self) -> None: ...
    def drain_badges(self) -> list[BadgeSignal]: ...
    @property
    def ground_truth(self) -> tuple[str, ...]: ...
    @property
    def description(self) -> str: ...


# ---------------------------------------------------------------------------
# Synthetic scene
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Actor:
    kind: str
    x: float
    y: float
    w: float
    h: float
    vx: float
    vy: float = 0.0
    phase: float = 0.0
    born: float = 0.0
    stop_at: float | None = None
    badge: bool | None = None
    tag: str = ""
    anomaly: str | None = None
    age: float = 0.0
    lifespan: float = 90.0

    @property
    def alive(self) -> bool:
        return -0.25 < self.x < 1.25 and self.age < self.lifespan


class SyntheticSource:
    """Deterministic, plausible CCTV scene with injectable anomalies."""

    def __init__(self, scenario: str = "lobby", *, width: int = 320, height: int = 180,
                 fps: float = 10.0, seed: int = 0, traffic: float = 1.0) -> None:
        self.scenario = scenario if scenario in _SCENE_SPAWN else "lobby"
        self.width = width
        self.height = height
        self.fps = fps
        self.traffic = max(0.05, traffic)
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._noise_tiles: list[np.ndarray] = []
        self._noise_index = 0
        self._t = 0.0
        self._actors: list[Actor] = []
        self._badges: list[BadgeSignal] = []
        self._active: dict[str, float] = {}
        self._spawn_timer = 0.8
        self._background = self._build_background()
        self._tag_counter = 100 + (seed * 37) % 800
        for _ in range(12):
            self._noise_tiles.append(self._np_rng.normal(0.0, 2.8, (height, width)).astype(np.int16))

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    @property
    def description(self) -> str:
        return f"synthetic:{self.scenario} ({self.width}x{self.height} @ {self.fps:g}fps)"

    @property
    def can_preroll(self) -> bool:
        return True

    @property
    def ground_truth(self) -> tuple[str, ...]:
        # A label is active only while the injected actor is still visible.
        # Without this, a runner who crossed the frame in three seconds left
        # five seconds of "anomalous" ground truth containing only normal
        # traffic, which unfairly punished the detector.
        living = {actor.anomaly for actor in self._actors if actor.anomaly and actor.alive}
        return tuple(
            kind for kind, expiry in self._active.items()
            if expiry > self._t and kind in living
        )

    def drain_badges(self) -> list[BadgeSignal]:
        badges, self._badges = self._badges, []
        return badges

    # -- rendering helpers -------------------------------------------------
    def _build_background(self) -> np.ndarray:
        h, w = self.height, self.width
        yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        img = 30.0 + 62.0 * yy + np.zeros((1, w), dtype=np.float32)
        if self.scenario == "parking":
            img = 18.0 + 44.0 * yy + np.zeros((1, w), dtype=np.float32)
        frame = np.clip(img, 0, 255).astype(np.uint8)

        if self.scenario in {"lobby", "office"}:
            # Rear wall and a bright doorway.
            frame[: int(h * 0.34)] = 46
            cv2.rectangle(frame, (int(w * 0.78), int(h * 0.18)), (int(w * 0.95), int(h * 0.72)), 190, -1)
            cv2.rectangle(frame, (int(w * 0.80), int(h * 0.22)), (int(w * 0.93), int(h * 0.68)), 205, -1)
            # Floor tiles.
            for i in range(1, 5):
                y = int(h * (0.38 + i * 0.12))
                cv2.line(frame, (0, y), (w, y), 82, 1)
            # Reception desk / furniture.
            cv2.rectangle(frame, (int(w * 0.03), int(h * 0.55)), (int(w * 0.24), int(h * 0.70)), 96, -1)
            cv2.rectangle(frame, (int(w * 0.03), int(h * 0.53)), (int(w * 0.24), int(h * 0.56)), 128, -1)
            if self.scenario == "office":
                for i in range(2):
                    x = int(w * (0.42 + i * 0.20))
                    cv2.rectangle(frame, (x, int(h * 0.58)), (x + int(w * 0.12), int(h * 0.64)), 88, -1)
        elif self.scenario == "server":
            frame[: int(h * 0.30)] = 38
            for i in range(4):
                x = int(w * (0.03 + i * 0.075))
                cv2.rectangle(frame, (x, int(h * 0.30)), (x + int(w * 0.05), int(h * 0.86)), 78, -1)
                cv2.rectangle(frame, (x + 2, int(h * 0.34)), (x + int(w * 0.05) - 2, int(h * 0.82)), 62, -1)
                for led in range(5):
                    ly = int(h * (0.37 + led * 0.08))
                    cv2.circle(frame, (x + int(w * 0.02), ly), 1, 210, -1)
            for i in range(4):
                x = int(w * (0.70 + i * 0.075))
                cv2.rectangle(frame, (x, int(h * 0.30)), (x + int(w * 0.05), int(h * 0.86)), 78, -1)
            # Aisle floor.
            cv2.rectangle(frame, (int(w * 0.30), int(h * 0.86)), (int(w * 0.70), int(h * 0.90)), 100, -1)
        elif self.scenario == "loading":
            frame[: int(h * 0.30)] = 42
            cv2.rectangle(frame, (int(w * 0.02), int(h * 0.14)), (int(w * 0.24), int(h * 0.60)), 150, -1)
            for i in range(1, 6):
                y = int(h * 0.14 + i * (h * 0.46 / 6))
                cv2.line(frame, (int(w * 0.02), y), (int(w * 0.24), y), 120, 1)
            for i in range(3):
                x = int(w * (0.62 + i * 0.11))
                cv2.rectangle(frame, (x, int(h * 0.52)), (x + int(w * 0.08), int(h * 0.72)), 104, -1)
        elif self.scenario == "parking":
            frame[: int(h * 0.30)] = 26
            # Painted bays.
            for i in range(1, 6):
                x = int(w * i / 6)
                cv2.line(frame, (x, int(h * 0.32)), (x, int(h * 0.78)), 120, 1)
            for i in range(4):
                y = int(h * (0.46 + i * 0.11))
                cv2.line(frame, (0, y), (w, int(h * 0.36)), 58, 1)
            # Parked cars (static background).
            for i, x in enumerate((0.54, 0.84)):
                cx = int(w * x)
                cv2.rectangle(frame, (cx - 28, int(h * 0.52)), (cx + 28, int(h * 0.66)), 70 + i * 6, -1)
                cv2.rectangle(frame, (cx - 16, int(h * 0.47)), (cx + 16, int(h * 0.55)), 84, -1)
        elif self.scenario == "stairwell":
            frame[: int(h * 0.24)] = 34
            for i in range(5):
                y0 = int(h * (0.34 + i * 0.11))
                cv2.rectangle(frame, (int(w * (0.05 + i * 0.02)), y0), (int(w * (0.95 - i * 0.02)), y0 + 6), 96, -1)
            cv2.rectangle(frame, (int(w * 0.33), int(h * 0.28)), (int(w * 0.48), int(h * 0.80)), 64, -1)

        # Vignette + baked texture.
        xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
        ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
        vignette = 1.0 - 0.28 * np.clip(xs * xs + ys * ys, 0.0, 1.0)
        texture = self._np_rng.normal(0.0, 2.0, (h, w)).astype(np.float32)
        frame = np.clip(frame.astype(np.float32) * vignette + texture, 0, 255).astype(np.uint8)
        return frame

    def _next_tag(self) -> str:
        self._tag_counter += 1 + self._rng.randrange(3)
        return f"P{self._tag_counter % 1000:03d}"

    def _spawn_person(self, *, x: float | None = None, y: float | None = None,
                      speed: float | None = None, direction: int | None = None,
                      stop_at: float | None = None, anomaly: str | None = None,
                      badge: bool | None = None) -> Actor:
        rng = self._rng
        if direction is None:
            direction = 1 if rng.random() < 0.5 else -1
        if x is None:
            x = -0.06 if direction > 0 else 1.02
        if y is None:
            y = rng.uniform(*_WALK_BAND)
        if speed is None:
            speed = rng.uniform(0.065, 0.115)
        if badge is None:
            badge = rng.random() < 0.93
        actor = Actor(
            kind="person",
            x=x, y=y,
            w=rng.uniform(0.042, 0.058),
            h=rng.uniform(0.23, 0.28),
            vx=direction * speed,
            vy=rng.uniform(-0.012, 0.012),
            phase=rng.uniform(0, math.tau),
            born=self._t,
            stop_at=stop_at,
            badge=badge,
            tag=self._next_tag(),
            anomaly=anomaly,
        )
        self._actors.append(actor)
        if badge:
            self._badges.append(BadgeSignal(person=actor.tag, door=self.scenario, granted=True))
        return actor

    def _spawn_vehicle(self, *, x: float | None = None, speed: float | None = None,
                       direction: int | None = None, anomaly: str | None = None) -> Actor:
        rng = self._rng
        if direction is None:
            direction = 1 if rng.random() < 0.5 else -1
        if x is None:
            x = -0.18 if direction > 0 else 1.12
        if speed is None:
            speed = rng.uniform(0.24, 0.36)
        actor = Actor(
            kind="vehicle",
            x=x,
            y=rng.uniform(0.55, 0.68),
            w=rng.uniform(0.19, 0.24),
            h=rng.uniform(0.085, 0.11),
            vx=direction * speed,
            born=self._t,
            tag=self._next_tag(),
            anomaly=anomaly,
        )
        self._actors.append(actor)
        return actor

    def _spawn_normal(self) -> None:
        if self.scenario in {"parking", "loading"} and self._rng.random() < 0.5:
            self._spawn_vehicle()
        else:
            self._spawn_person()

    # -- anomaly injection -------------------------------------------------
    def inject(self, kind: str, duration: float = 14.0) -> bool:
        """Inject a labelled anomaly into the scene.

        Returns False for unknown kinds instead of raising so the console can
        offer a best-effort demo action.
        """

        if kind not in ANOMALY_KINDS:
            return False
        self._active[kind] = self._t + duration
        before = len(self._actors)
        if kind == "running":
            self._spawn_person(speed=self._rng.uniform(0.34, 0.46), direction=1, x=-0.08, anomaly=kind)
        elif kind == "crowd":
            for i in range(7):
                self._spawn_person(
                    x=0.10 + i * 0.105,
                    y=self._rng.uniform(0.42, 0.72),
                    speed=self._rng.uniform(0.012, 0.035),
                    anomaly=kind,
                    badge=True,
                )
        elif kind == "loitering":
            self._spawn_person(
                x=self._rng.uniform(0.30, 0.55),
                direction=1,
                speed=self._rng.uniform(0.05, 0.08),
                stop_at=self._rng.uniform(0.34, 0.60),
                anomaly=kind,
            )
        elif kind == "abandoned":
            self._actors.append(
                Actor(
                    kind="object",
                    x=0.42, y=0.62,
                    w=0.075, h=0.085,
                    vx=0.0,
                    born=self._t,
                    anomaly=kind,
                    tag="OBJ",
                )
            )
        elif kind == "vehicle":
            self._spawn_vehicle(speed=self._rng.uniform(0.20, 0.28), anomaly=kind)
        elif kind == "tailgate":
            tag = self._next_tag()
            # One badge for two people entering together: the classic tailgate.
            first = self._spawn_person(x=0.82, y=0.56, direction=-1, speed=0.09,
                                       anomaly=kind, badge=True)
            first.tag = tag
            second = self._spawn_person(x=0.86, y=0.60, direction=-1, speed=0.085,
                                        anomaly=kind, badge=False)
            second.tag = tag
        elif kind == "after_hours":
            self._spawn_person(speed=self._rng.uniform(0.07, 0.10), anomaly=kind)
        for actor in self._actors[before:]:
            # Keep the scene and the ground-truth label aligned: an injected
            # actor must not linger long after its anomaly window closes, or
            # the detector is penalised for correctly noticing it.
            actor.lifespan = min(actor.lifespan, duration)
        return True

    # -- frame production --------------------------------------------------
    def read(self) -> np.ndarray | None:
        dt = 1.0 / self.fps
        self._t += dt

        # Spawn normal traffic.
        self._spawn_timer -= dt * self.traffic
        if self._spawn_timer <= 0:
            self._spawn_normal()
            lo, hi = _SCENE_SPAWN[self.scenario]
            self._spawn_timer = self._rng.uniform(lo, hi) / self.traffic

        # Advance actors.
        for actor in self._actors:
            actor.age += dt
            if actor.stop_at is not None and abs(actor.x - actor.stop_at) < 0.02:
                actor.vx = 0.0
                actor.vy = 0.0
            actor.x += actor.vx * dt
            actor.y += actor.vy * dt
            actor.phase += dt * (9.0 if actor.kind == "person" else 3.0)
        self._actors = [a for a in self._actors if a.alive][:80]

        # Compose the frame: baked background + sensor noise + actors.
        tile = self._noise_tiles[self._noise_index]
        self._noise_index = (self._noise_index + 1) % len(self._noise_tiles)
        frame = np.clip(self._background.astype(np.int16) + tile, 0, 255).astype(np.uint8)
        for actor in self._actors:
            self._draw_actor(frame, actor)
        return frame

    def _draw_actor(self, frame: np.ndarray, actor: Actor) -> None:
        h, w = self.height, self.width
        x = int(actor.x * w)
        y = int(actor.y * h)
        bw = max(2, int(actor.w * w))
        bh = max(2, int(actor.h * h))

        if actor.kind == "person":
            cx = x + bw // 2
            body_top = y + int(bh * 0.26)
            # Contact shadow.
            cv2.ellipse(frame, (cx, y + bh), (max(3, bw // 2), max(1, bh // 14)), 0, 0, 360, 18, -1)
            # Torso.
            cv2.ellipse(frame, (cx, body_top + int(bh * 0.24)),
                        (max(2, bw // 2), max(3, int(bh * 0.24))), 0, 0, 360, 196, -1)
            # Head.
            cv2.circle(frame, (cx, y + int(bh * 0.14)), max(2, int(bw * 0.42)), 214, -1)
            # Legs (animated gait).
            swing = int(math.sin(actor.phase) * bh * 0.06)
            cv2.line(frame, (cx - bw // 5, body_top + int(bh * 0.42)),
                     (cx - bw // 5 + swing, y + bh), 176, max(1, bw // 5))
            cv2.line(frame, (cx + bw // 5, body_top + int(bh * 0.42)),
                     (cx + bw // 5 - swing, y + bh), 176, max(1, bw // 5))
        elif actor.kind == "vehicle":
            cv2.ellipse(frame, (x + bw // 2, y + bh), (bw // 2, max(1, bh // 5)), 0, 0, 360, 16, -1)
            cv2.rectangle(frame, (x, y + bh // 3), (x + bw, y + bh), 168, -1)
            cv2.rectangle(frame, (x + bw // 4, y), (x + int(bw * 0.72), y + bh // 2), 132, -1)
            # Headlights / windscreen.
            cv2.circle(frame, (x + 3, y + int(bh * 0.70)), 2, 250, -1)
            cv2.circle(frame, (x + bw - 3, y + int(bh * 0.70)), 2, 250, -1)
            cv2.rectangle(frame, (x + int(bw * 0.36), y + 2), (x + int(bw * 0.62), y + bh // 3), 96, -1)
        else:  # object
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), 228, -1)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), 120, max(1, bw // 10))
            cv2.line(frame, (x, y), (x + bw, y + bh), 150, 1)


# ---------------------------------------------------------------------------
# Real inputs
# ---------------------------------------------------------------------------

class VideoSource:
    """Webcam, file or RTSP source, decoded with OpenCV."""

    def __init__(self, spec: str, *, fps: float = 10.0, max_width: int = 480) -> None:
        self.spec = spec
        self.fps = fps
        self.width = max_width
        self.height = int(max_width * 9 / 16)
        self._cap: cv2.VideoCapture | None = None
        self._max_width = max_width
        self._is_file = spec.startswith("video:")
        self._path = spec.split(":", 1)[1] if ":" in spec else spec

    def open(self) -> None:
        target: int | str
        if self.spec.startswith("webcam:"):
            target = int(self.spec.split(":", 1)[1])
        elif self.spec.startswith("video:"):
            target = self._path
            if not Path(target).exists():
                raise FileNotFoundError(f"video file not found: {target}")
        else:
            target = self.spec
        backend = cv2.CAP_FFMPEG if isinstance(target, str) and target.startswith("rtsp") else cv2.CAP_ANY
        cap = cv2.VideoCapture(target, backend)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video source: {self.spec}")
        self._cap = cap
        ok, frame = cap.read()
        if ok and frame is not None:
            self.height, self.width = frame.shape[:2]
            if self.width > self._max_width:
                scale = self._max_width / self.width
                self.width = self._max_width
                self.height = int(self.height * scale)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def description(self) -> str:
        return self.spec

    @property
    def can_preroll(self) -> bool:
        # Files can be fast-forwarded; a live webcam/RTSP feed cannot.
        return self._is_file

    @property
    def ground_truth(self) -> tuple[str, ...]:
        return ()

    def drain_badges(self) -> list[BadgeSignal]:
        return []

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            if self._is_file:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
            if not ok or frame is None:
                return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape[1] > self._max_width:
            scale = self._max_width / gray.shape[1]
            gray = cv2.resize(gray, (self._max_width, int(gray.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        self.height, self.width = gray.shape[:2]
        return gray


def resolve_source(camera: CameraConfig) -> FrameSource:
    spec = camera.source.strip()
    if spec.startswith("synthetic:"):
        scenario = spec.split(":", 1)[1] or "lobby"
        return SyntheticSource(
            scenario, fps=camera.fps, seed=camera.seed, traffic=camera.traffic
        )
    if spec.startswith(("webcam:", "video:", "rtsp://", "http://", "https://")):
        return VideoSource(spec, fps=camera.fps)
    # Bare path -> video file.
    return VideoSource(f"video:{spec}", fps=camera.fps)


@dataclass(slots=True)
class DemoScript:
    """Scheduled anomaly injections used by --demo and the benchmark."""

    steps: list[tuple[float, str, str | None]] = field(default_factory=list)
    duration: float = 0.0
    anomaly_seconds: float = 14.0

    @classmethod
    def standard(cls, warmup: float = 16.0) -> "DemoScript":
        steps = [
            (warmup + 2.0, "running", "CAM-03"),
            (warmup + 18.0, "crowd", "CAM-06"),
            (warmup + 34.0, "tailgate", "CAM-01"),
            (warmup + 47.0, "loitering", "CAM-02"),
            (warmup + 60.0, "abandoned", "CAM-05"),
            (warmup + 73.0, "vehicle", "CAM-08"),
            (warmup + 86.0, "after_hours", "CAM-04"),
        ]
        return cls(steps=steps, duration=steps[-1][0] + 18.0)

    @classmethod
    def stress(cls, warmup: float = 16.0) -> "DemoScript":
        """Dense injection schedule for demos that want constant action."""
        steps: list[tuple[float, str, str | None]] = []
        cameras = ["CAM-01", "CAM-02", "CAM-03", "CAM-04", "CAM-05", "CAM-06", "CAM-07", "CAM-08"]
        kinds = ["running", "crowd", "loitering", "abandoned", "tailgate", "vehicle", "after_hours"]
        t = warmup + 2.0
        for i in range(28):
            steps.append((t, kinds[i % len(kinds)], cameras[(i * 3) % len(cameras)]))
            t += 7.0
        return cls(steps=steps, duration=t + 10.0, anomaly_seconds=14.0)

    @classmethod
    def evaluation(cls, warmup: float = 10.0, interval: float = 10.0,
                   anomaly_seconds: float = 8.0) -> "DemoScript":
        """Isolated anomalies used by the benchmark.

        Only *sequence-detectable* anomalies are scheduled here: running,
        crowd formation and after-hours motion.  Semantic events (tailgates,
        loitering, abandoned objects, vehicles in restricted zones) are handled
        by explicit correlation rules and are covered by unit tests instead of
        being dishonestly folded into a reconstruction-error AUROC.

        Each camera receives two well-separated anomalies.  The gaps are long
        enough that a 24-frame feature window contains *either* an anomaly or
        normal operation, never both, which makes window-level AUROC meaningful.
        """
        schedule = [
            ("running", "CAM-01"), ("crowd", "CAM-06"),
            ("after_hours", "CAM-03"), ("running", "CAM-02"),
            ("crowd", "CAM-05"), ("after_hours", "CAM-04"),
            ("running", "CAM-07"), ("after_hours", "CAM-08"),
            ("crowd", "CAM-01"), ("running", "CAM-06"),
            ("after_hours", "CAM-03"), ("running", "CAM-05"),
            ("crowd", "CAM-02"), ("running", "CAM-04"),
            ("after_hours", "CAM-07"), ("running", "CAM-08"),
        ]
        steps = [
            (warmup + index * interval, kind, camera)
            for index, (kind, camera) in enumerate(schedule)
        ]
        return cls(
            steps=steps,
            duration=warmup + len(steps) * interval + 10.0,
            anomaly_seconds=anomaly_seconds,
        )
