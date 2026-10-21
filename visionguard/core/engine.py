"""Pipeline orchestration.

One background thread owns all camera pipelines (detector, tracker, feature
extractor, anomaly scorer, correlator), persists events and publishes an
immutable :class:`~visionguard.core.types.Snapshot` that the console can read
without locks.

Synthetic scenes and video files are **pre-rolled** at start-up: frames are
processed as fast as the CPU allows, with simulated timestamps, so the
autoencoder calibrates on ~90 seconds of scene behaviour before the operator
sees the dashboard.  Live cameras (webcam / RTSP) cannot be pre-rolled; they
calibrate in real time and the console shows progress.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from visionguard.config import AppConfig, CameraConfig, RuleConfig, resolve_path
from visionguard.core.store import EventStore
from visionguard.core.types import (
    AnomalyResult,
    BadgeEvent,
    BBox,
    CameraState,
    CameraView,
    Event,
    MetricsView,
    Severity,
    Snapshot,
    TrackView,
)
from visionguard.correlation.rules import CorrelationEngine
from visionguard.vision.anomaly import AnomalyScorer
from visionguard.vision.detector import MotionDetector, Tracker, frame_energy
from visionguard.vision.features import CameraFeatures, FeatureExtractor, track_views
from visionguard.vision.source import DemoScript, FrameSource, SyntheticSource, VideoSource, resolve_source

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Sample:
    """One scored window, retained for benchmarking."""

    camera_id: str
    ts: float
    score: float
    error: float
    threshold: float
    flagged: bool
    ground_truth: tuple[str, ...]
    persons: int


@dataclass(slots=True)
class Runtime:
    config: CameraConfig
    source: FrameSource
    detector: MotionDetector
    tracker: Tracker
    extractor: FeatureExtractor
    scorer: AnomalyScorer
    correlator: CorrelationEngine
    frame: np.ndarray | None = None
    prev_frame: np.ndarray | None = None
    features: CameraFeatures | None = None
    result: AnomalyResult = field(default_factory=lambda: AnomalyResult.warmup(0.0))
    fps: float = 0.0
    latency_ms: float = 0.0
    state: CameraState = CameraState.OFFLINE
    last_event: str | None = None
    frames_seen: int = 0
    last_frame_sim: float = -1e9
    last_wall: float = 0.0
    next_due: float = 0.0
    last_windows: int = 0
    last_recorded: int = -1
    stalled: bool = False
    stall_reported: float = -1e9
    gt_history: deque[tuple[float, tuple[str, ...]]] = field(default_factory=lambda: deque(maxlen=400))
    samples: list[Sample] = field(default_factory=list)


class Engine:
    def __init__(
        self,
        config: AppConfig,
        *,
        demo_script: DemoScript | None = None,
        clock_epoch: float | None = None,
        speed: float = 1.0,
        preroll: bool = True,
        record: bool = False,
        root: str | Path | None = None,
        site: str = "Intelligent CCTV Monitoring System",
    ) -> None:
        self.config = config.validate()
        self.demo_script = demo_script
        self.speed = max(0.1, float(speed))
        self.preroll = preroll
        self.record = record
        self.root = Path(root or Path.cwd())
        self.site = site
        self.binding = "0.0.0.0"

        now = time.time()
        if clock_epoch is None:
            if demo_script is not None:
                # Match the original mock-up's 02:14:33 night-shift scenario.
                base = datetime.now().replace(hour=2, minute=14, second=33, microsecond=0)
            else:
                base = datetime.now()
            clock_epoch = base.timestamp()
        self.clock_epoch = clock_epoch

        self.db_path = resolve_path(self.root, config.db_path)
        self.model_dir = resolve_path(self.root, config.model_dir)
        self.snapshot_dir = resolve_path(self.root, config.snapshot_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self._store: EventStore | None = None
        self._runtimes: list[Runtime] = []
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._snapshot: Snapshot | None = None
        self._events: deque[Event] = deque(maxlen=config.retention_events)
        self._next_id = 1
        self._started_at = time.monotonic()
        self._error: str | None = None
        self._stage = "starting"
        self._stage_progress = 0.0
        self._demo_index = 0
        self._paused = False
        self._pause_started = 0.0
        self._paused_total = 0.0
        self._det_buckets: Counter[int] = Counter()
        self._detections = 0
        self._frames = 0
        self._windows = 0
        self._event_counts = {"critical": 0, "warning": 0, "info": 0}
        self._sim = 0.0
        self._last_publish = 0.0
        self._last_stall_check = 0.0
        self._recorded: list[Sample] = []
        self._last_error_log = 0.0
        self.backend = "pending"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="visionguard-engine", daemon=True)
        self._thread.start()

    def wait_ready(self, timeout: float = 30.0) -> bool:
        return self._ready.wait(timeout)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        for runtime in self._runtimes:
            try:
                runtime.source.close()
            except Exception:  # pragma: no cover - best effort
                pass
        if self._store is not None:
            try:
                self._store.close()
            except Exception:  # pragma: no cover
                pass
            self._store = None

    def snapshot(self) -> Snapshot | None:
        return self._snapshot

    def status(self) -> dict[str, Any]:
        return {
            "stage": self._stage,
            "progress": self._stage_progress,
            "ready": self._ready.is_set(),
            "error": self._error,
        }

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            if paused and not self._paused:
                self._paused = True
                self._pause_started = time.monotonic()
            elif not paused and self._paused:
                self._paused = False
                self._paused_total += time.monotonic() - self._pause_started

    @property
    def paused(self) -> bool:
        return self._paused

    def inject(self, kind: str, camera_id: str | None = None) -> tuple[bool, str]:
        runtime = None
        if camera_id is None:
            runtime = self._runtimes[0] if self._runtimes else None
        else:
            runtime = next((r for r in self._runtimes if r.config.id == camera_id), None)
        if runtime is None:
            return False, "no such camera"
        source = runtime.source
        if not isinstance(source, SyntheticSource):
            return False, "camera is not a synthetic scene"
        ok = source.inject(kind)
        return ok, f"{kind} injected into {runtime.config.id}" if ok else f"unknown anomaly: {kind}"

    def acknowledge(self, event_id: int) -> bool:
        if self._store is None:
            return False
        ok = self._store.acknowledge(event_id)
        if ok:
            with self._lock:
                for event in self._events:
                    if event.id == event_id:
                        event.ack = True
                        break
        return ok

    def save_snapshot(self, camera_id: str) -> Path | None:
        runtime = next((r for r in self._runtimes if r.config.id == camera_id), None)
        if runtime is None or runtime.frame is None:
            return None
        image = cv2.cvtColor(runtime.frame, cv2.COLOR_GRAY2BGR)
        h, w = image.shape[:2]
        for track in runtime.tracker.tracks:
            if not track.confirmed:
                continue
            x0 = int(track.bbox.x * w)
            y0 = int(track.bbox.y * h)
            x1 = int((track.bbox.x + track.bbox.w) * w)
            y1 = int((track.bbox.y + track.bbox.h) * h)
            colour = (0, 0, 255) if track.label == "person" else (0, 220, 255)
            cv2.rectangle(image, (x0, y0), (x1, y1), colour, 1)
            cv2.putText(image, f"{track.label} {track.id}", (x0, max(10, y0 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, colour, 1, cv2.LINE_AA)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = self.snapshot_dir / f"{camera_id}-{stamp}.png"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)
        return path

    def recorded_samples(self) -> list[Sample]:
        with self._lock:
            samples = list(self._recorded)
        for runtime in self._runtimes:
            samples.extend(runtime.samples)
        samples.sort(key=lambda s: (s.camera_id, s.ts))
        return samples

    def event_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._event_counts)

    def metrics(self) -> MetricsView:
        snapshot = self._snapshot
        if snapshot is not None:
            return snapshot.metrics
        return self._build_metrics()

    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------
    def _run(self) -> None:
        try:
            self._setup()
            self._loop()
        except Exception as exc:  # pragma: no cover - startup failures
            log.exception("engine failed")
            self._error = f"{type(exc).__name__}: {exc}"
            self._stage = "failed"
            self._ready.set()
        finally:
            try:
                for runtime in self._runtimes:
                    runtime.source.close()
            except Exception:
                pass

    def _setup(self) -> None:
        self._stage = "opening event store"
        self._store = EventStore(self.db_path)

        cameras = self.config.cameras
        for index, camera in enumerate(cameras):
            self._stage = f"opening {camera.id}"
            self._stage_progress = index / max(1, len(cameras))
            runtime = self._create_runtime(camera)
            try:
                runtime.source.open()
            except Exception as exc:
                log.warning("%s: source unavailable: %s", camera.id, exc)
                runtime.state = CameraState.OFFLINE
            self._runtimes.append(runtime)

        if self.preroll:
            total = len(self._runtimes)
            workers = min(4, max(1, total))
            done = 0
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="viq-preroll") as pool:
                futures = {pool.submit(self._preroll, runtime): runtime for runtime in self._runtimes}
                for future in as_completed(futures):
                    runtime = futures[future]
                    done += 1
                    self._stage = f"calibrating ({done}/{total})"
                    self._stage_progress = done / max(1, total)
                    try:
                        future.result()
                    except Exception as exc:  # pragma: no cover - per-camera isolation
                        log.warning("%s: preroll failed: %s", runtime.config.id, exc)

        for runtime in self._runtimes:
            runtime.correlator = CorrelationEngine(runtime.config, self.config.rules)
            runtime.tracker.reset()
            runtime.next_due = 0.0
            # Pre-roll ran at CPU speed; do not report that throughput as live
            # inference rate, and do not show a warmed camera as offline in the
            # first snapshot (published just before the first live frame).
            runtime.last_wall = 0.0
            runtime.fps = 0.0
            if runtime.frames_seen:
                runtime.state = (
                    CameraState.LIVE if runtime.scorer.calibrated else CameraState.CALIBRATING
                )

        backends = {r.scorer.backend_name for r in self._runtimes}
        self.backend = backends.pop() if len(backends) == 1 else "mixed"
        self._stage = "live"
        self._stage_progress = 1.0
        self._ready.set()
        self._publish_snapshot()

    def _create_runtime(self, camera: CameraConfig) -> Runtime:
        source = resolve_source(camera)
        model_path = self.model_dir / f"{camera.id}.npz"
        scorer = AnomalyScorer(
            self.config.model,
            camera.id,
            seed=camera.seed or 1,
            model_path=model_path,
        )
        return Runtime(
            config=camera,
            source=source,
            detector=MotionDetector(),
            tracker=Tracker(),
            extractor=FeatureExtractor(),
            scorer=scorer,
            correlator=CorrelationEngine(camera, self.config.rules),
        )

    def _preroll(self, runtime: Runtime) -> None:
        if not getattr(runtime.source, "can_preroll", False):
            return
        model = self.config.model
        if runtime.scorer.calibrated:
            # A persisted model is available; only warm the background model.
            frames = min(80, (model.window + 4) * model.stride)
        else:
            frames = (model.window + model.warmup_windows + 8) * model.stride
        fps = max(runtime.config.fps, 1.0)
        started = time.monotonic()
        for i in range(frames):
            ts = -(frames - i) / fps
            self._process(runtime, ts, emit=False)
            if time.monotonic() - started > 45.0:
                log.warning("%s: preroll stopped after 45s", runtime.config.id)
                break
        log.info("%s: pre-rolled %d frames, threshold=%.4f", runtime.config.id,
                 runtime.frames_seen, runtime.scorer.threshold)

    def _loop(self) -> None:
        origin = time.monotonic()
        while not self._stop.is_set():
            if self._paused:
                time.sleep(0.05)
                continue
            wall = time.monotonic() - origin - self._paused_total
            sim = wall * self.speed
            self._sim = sim
            clock = self.clock_epoch + sim

            for runtime in self._runtimes:
                if sim + 1e-9 < runtime.next_due:
                    continue
                interval = 1.0 / runtime.config.fps
                runtime.next_due += interval
                if runtime.next_due < sim - 1.0:
                    runtime.next_due = sim + interval
                self._process(runtime, clock, emit=True)

            self._run_demo(sim)

            if sim - self._last_stall_check >= 1.0:
                self._last_stall_check = sim
                self._check_stalls(clock, sim)

            if sim - self._last_publish >= 0.1:
                self._last_publish = sim
                self._publish_snapshot()

            time.sleep(0.002)

        self._publish_snapshot()

    def _run_demo(self, sim: float) -> None:
        if self.demo_script is None:
            return
        while self._demo_index < len(self.demo_script.steps):
            at, kind, camera_id = self.demo_script.steps[self._demo_index]
            if sim < at:
                break
            self._demo_index += 1
            runtime = next((r for r in self._runtimes if r.config.id == camera_id), None)
            source = runtime.source if runtime is not None else None
            duration = self.demo_script.anomaly_seconds
            if isinstance(source, SyntheticSource):
                ok = source.inject(kind, duration=duration)
                message = f"{kind} injected into {runtime.config.id}" if ok else f"unknown anomaly: {kind}"
            else:
                ok, message = self.inject(kind, camera_id)
            log.info("demo: %s", message)
            if not ok:
                log.warning("demo injection failed: %s", message)

    def _check_stalls(self, clock: float, sim: float) -> None:
        for runtime in self._runtimes:
            if runtime.frames_seen == 0:
                continue
            stale = sim - runtime.last_frame_sim
            if stale > 5.0:
                if runtime.state is not CameraState.OFFLINE:
                    runtime.state = CameraState.OFFLINE
                    runtime.stalled = True
                if self.config.rules.has("source_stall") and sim - runtime.stall_reported > 120.0:
                    self._emit(
                        Event(
                            id=0,
                            ts=clock,
                            camera_id=runtime.config.id,
                            camera_name=runtime.config.name,
                            zone=runtime.config.zone,
                            kind="source_stall",
                            severity=Severity.CRITICAL,
                            message=f"No frames for {stale:.0f}s from {runtime.source.description}",
                            score=1.0,
                            detail={"stale_s": round(stale, 1)},
                        )
                    )
                    runtime.stall_reported = sim
            elif runtime.stalled:
                runtime.stalled = False
                log.info("%s: stream recovered", runtime.config.id)

    # ------------------------------------------------------------------
    # Per-frame pipeline
    # ------------------------------------------------------------------
    def _process(self, runtime: Runtime, ts: float, *, emit: bool) -> None:
        wall_start = time.perf_counter()
        try:
            frame = runtime.source.read()
        except Exception as exc:  # pragma: no cover - camera/network failures
            if time.monotonic() - self._last_error_log > 30.0:
                log.error("%s: source read failed: %s", runtime.config.id, exc)
                self._last_error_log = time.monotonic()
            return
        if frame is None:
            return
        runtime.frames_seen += 1
        runtime.last_frame_sim = self._sim

        detections = runtime.detector.detect(
            frame, protect=[track.bbox for track in runtime.tracker.tracks]
        )
        tracks = runtime.tracker.update(detections, ts)
        energy = frame_energy(runtime.prev_frame, frame)
        runtime.prev_frame = frame
        features = runtime.extractor.extract(tracks, energy, ts)
        result = runtime.scorer.observe(features.values)
        runtime.features = features
        runtime.result = result
        runtime.frame = self._display_frame(frame)

        elapsed_ms = (time.perf_counter() - wall_start) * 1000.0
        runtime.latency_ms = elapsed_ms if runtime.latency_ms == 0 else 0.9 * runtime.latency_ms + 0.1 * elapsed_ms

        now_wall = time.perf_counter()
        if runtime.last_wall > 0:
            frame_fps = 1.0 / max(1e-3, now_wall - runtime.last_wall)
            runtime.fps = frame_fps if runtime.fps == 0 else 0.85 * runtime.fps + 0.15 * frame_fps
        runtime.last_wall = now_wall

        with self._lock:
            if emit:
                # Preroll is calibration, not live telemetry.  Counting it
                # here used to make the first dashboard frame report hundreds
                # of frames and a fictional detections/hour figure.
                self._frames += 1
                self._detections += len(detections)
                if not result.calibrating and result.windows != runtime.last_windows:
                    self._windows += 1
                self._det_buckets[int(self._sim // 60)] += len(detections)
        runtime.last_windows = result.windows

        if not emit:
            runtime.source.drain_badges()
            runtime.gt_history.append((ts, runtime.source.ground_truth))
            return

        runtime.gt_history.append((ts, runtime.source.ground_truth))
        badges = [
            BadgeEvent(ts=ts, person=badge.person, door=badge.door, granted=badge.granted)
            for badge in runtime.source.drain_badges()
        ]
        for event in runtime.correlator.observe(ts, tracks, features, result, badges):
            self._emit(event)

        if self.record and not result.calibrating and result.windows != runtime.last_recorded:
            runtime.last_recorded = result.windows
            window_seconds = self.config.model.window * self.config.model.stride / max(runtime.config.fps, 1.0)
            labels: set[str] = set()
            for sample_ts, gt in runtime.gt_history:
                if sample_ts >= ts - window_seconds:
                    labels.update(gt)
            runtime.samples.append(
                Sample(
                    camera_id=runtime.config.id,
                    ts=ts,
                    score=result.score,
                    error=result.raw_error,
                    threshold=result.threshold,
                    flagged=result.is_anomaly,
                    ground_truth=tuple(sorted(labels)),
                    persons=features.persons,
                )
            )
            if len(runtime.samples) > 20000:
                del runtime.samples[:5000]
        self._update_state(runtime, ts)

    @staticmethod
    def _index_windows(result: AnomalyResult) -> int:
        return result.windows

    def _display_frame(self, frame: np.ndarray) -> np.ndarray:
        target_w = self.config.display_width
        target_h = self.config.display_height
        if frame.shape[1] <= target_w and frame.shape[0] <= target_h:
            return frame
        scale = min(target_w / frame.shape[1], target_h / frame.shape[0])
        size = (max(2, int(frame.shape[1] * scale)), max(2, int(frame.shape[0] * scale)))
        return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

    def _update_state(self, runtime: Runtime, ts: float) -> None:
        result = runtime.result
        if runtime.frames_seen == 0:
            state = CameraState.OFFLINE
        elif result.calibrating:
            state = CameraState.CALIBRATING
        elif result.is_anomaly or result.score > 0.85:
            state = CameraState.ALERT
        elif result.score > 0.65:
            state = CameraState.WARN
        else:
            state = CameraState.LIVE
        runtime.state = state

    # ------------------------------------------------------------------
    # Events and snapshots
    # ------------------------------------------------------------------
    def _emit(self, event: Event) -> None:
        with self._lock:
            event.id = self._next_id
            self._next_id += 1
            self._events.appendleft(event)
            self._event_counts[event.severity.value] = self._event_counts.get(event.severity.value, 0) + 1
            runtime = next((r for r in self._runtimes if r.config.id == event.camera_id), None)
            if runtime is not None:
                runtime.last_event = event.kind
            if self._store is not None:
                try:
                    stored_id = self._store.insert(event)
                    event.id = stored_id
                    self._next_id = max(self._next_id, stored_id + 1)
                except Exception as exc:  # pragma: no cover - disk failures
                    log.error("could not persist event: %s", exc)

    def _publish_snapshot(self) -> None:
        minute = int(self._sim // 60)
        self._det_buckets = Counter(
            {bucket: count for bucket, count in self._det_buckets.items() if bucket >= minute - 10}
        )
        with self._lock:
            events = tuple(self._events)
            retention = self.config.rules.alert_retention_minutes * 60.0
            now_epoch = self.clock_epoch + self._sim
            active = sum(
                1 for event in events
                if not event.ack and event.severity.rank >= 1 and now_epoch - event.ts <= retention
            )
            cameras = tuple(self._camera_view(runtime) for runtime in self._runtimes)
            metrics = self._build_metrics(active=active)
            health = {
                "cameras": f"{sum(1 for c in cameras if c.state != CameraState.OFFLINE)}/{len(cameras)} online",
                "model": self.backend,
                "store": "sqlite",
                "acam": "correlated" if any(c.access_controlled for c in self.config.cameras) else "off",
                "pipeline": "degraded" if self._error else "ok",
            }

        snapshot = Snapshot(
            ts=self.clock_epoch + self._sim,
            clock=time.strftime("%H:%M:%S", time.localtime(self.clock_epoch + self._sim)),
            cameras=cameras,
            events=events,
            metrics=metrics,
            health=health,
            paused=self._paused,
            demo=self.demo_script is not None,
            model=self.backend,
            site=self.site,
        )
        self._snapshot = snapshot

    def _build_metrics(self, active: int = 0) -> MetricsView:
        uptime = max(1.0, time.monotonic() - self._started_at)
        minute = int(self._sim // 60)
        buckets = [self._det_buckets.get(minute - i, 0) for i in range(10)]
        window_minutes = min(10.0, max(self._sim / 60.0, 0.5))
        rate = sum(buckets) / window_minutes * 60.0
        latencies = [r.latency_ms for r in self._runtimes if r.latency_ms > 0]
        fps = sum(r.fps for r in self._runtimes)
        return MetricsView(
            fps=fps,
            detections=self._detections,
            detections_per_hour=rate,
            events=sum(self._event_counts.values()),
            critical=self._event_counts.get("critical", 0),
            warnings=self._event_counts.get("warning", 0),
            active_alerts=active,
            windows_scored=self._windows,
            uptime_s=uptime,
            latency_ms=float(np.mean(latencies)) if latencies else 0.0,
            frames=self._frames,
        )

    def _camera_view(self, runtime: Runtime) -> CameraView:
        score = runtime.result.score if not runtime.result.calibrating else 0.0
        return CameraView(
            id=runtime.config.id,
            name=runtime.config.name,
            zone=runtime.config.zone,
            state=runtime.state,
            score=score,
            threshold=runtime.result.threshold,
            raw_error=runtime.result.raw_error,
            calibrating=runtime.result.calibrating,
            progress=runtime.result.progress,
            persons=runtime.features.persons if runtime.features else 0,
            vehicles=runtime.features.vehicles if runtime.features else 0,
            objects=runtime.features.objects if runtime.features else 0,
            fps=runtime.fps,
            latency_ms=runtime.latency_ms,
            frame=runtime.frame,
            tracks=track_views(runtime.tracker.tracks),
            last_event=runtime.last_event,
            backend=runtime.scorer.backend_name,
            frames_seen=runtime.frames_seen,
            tint=runtime.config.tint,
        )
