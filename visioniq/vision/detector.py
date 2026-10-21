"""Foreground detection and multi-object tracking.

Detection uses a per-pixel adaptive background model rather than MOG2:

* the background is updated with a per-pixel learning rate;
* pixels inside a currently tracked object are *frozen* (learning rate 0).

Freezing is what makes this useful for security analysis.  A conventional
background subtractor absorbs a person who stands still within a few seconds,
so loitering and abandoned-object rules can never fire.  Here the background
never learns the object in the first place, so a stationary person or a
dropped bag stays foreground for as long as it is there, and it leaves no
ghost when it is gone.

Labels (person / vehicle / object) are geometric heuristics, not a learned
classifier.  A real deployment would replace this stage with a trained
detector; the rest of the pipeline consumes the same Detection/Track types, so
that replacement is isolated to this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from visioniq.core.types import BBox, Detection, Track

_TYPICAL_PERSON_AREA = 0.055 * 0.26


@dataclass(slots=True)
class DetectorConfig:
    min_area: float = 0.0012
    max_area: float = 0.55
    threshold: float = 20.0
    learning_rate: float = 0.02
    protect_margin: float = 0.08
    blur: int = 3


class MotionDetector:
    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self._background: np.ndarray | None = None
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def reset(self) -> None:
        self._background = None

    def detect(self, frame: np.ndarray, protect: Sequence[BBox] = ()) -> list[Detection]:
        if frame.ndim != 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        if self._background is None or self._background.shape != frame.shape:
            self._background = frame.astype(np.float32)
            return []

        background_u8 = np.clip(self._background, 0, 255).astype(np.uint8)
        diff = cv2.absdiff(frame, background_u8)
        if self.config.blur > 0:
            diff = cv2.GaussianBlur(diff, (self.config.blur, self.config.blur), 0)
        mask = (diff > self.config.threshold).astype(np.uint8) * 255

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.dilate(mask, self._kernel, iterations=2)

        # Per-pixel learning rate; zero where an object is being tracked.
        alpha = np.full((h, w), self.config.learning_rate, dtype=np.float32)
        margin = self.config.protect_margin
        for box in protect:
            # Absolute padding as well as a relative margin: detectors often
            # return a partial silhouette (e.g. torso without legs), and an
            # under-sized protection box would let the background absorb the
            # rest of the person, splitting the blob on the next frame.
            x0 = max(0, int((box.x - box.w * margin - 0.02) * w))
            y0 = max(0, int((box.y - box.h * margin - 0.04) * h))
            x1 = min(w, int((box.x + box.w * (1 + margin) + 0.02) * w) + 1)
            y1 = min(h, int((box.y + box.h * (1 + margin) + 0.04) * h) + 1)
            if x1 > x0 and y1 > y0:
                alpha[y0:y1, x0:x1] = 0.0
        self._background += alpha * (frame.astype(np.float32) - self._background)

        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        detections: list[Detection] = []
        total = float(w * h)
        for i in range(1, count):
            x, y, bw, bh, area = (int(v) for v in stats[i])
            if bw <= 2 or bh <= 2:
                continue
            area_frac = area / total
            if area_frac < self.config.min_area or area_frac > self.config.max_area:
                continue
            bbox = BBox(
                x=float(x) / w,
                y=float(y) / h,
                w=float(bw) / w,
                h=float(bh) / h,
            ).clipped()
            label, instances = self._classify(bbox, area_frac)
            confidence = min(1.0, 0.45 + area_frac / (self.config.min_area * 12.0))
            detections.append(Detection(bbox, label, confidence, instances))
        return detections

    @staticmethod
    def _classify(bbox: BBox, area_frac: float) -> tuple[str, int]:
        aspect = bbox.aspect
        width_ratio = bbox.w / max(bbox.h, 1e-6)
        if width_ratio >= 1.75 and bbox.w >= 0.05 and area_frac > 0.006:
            return "vehicle", 1
        if bbox.area > _TYPICAL_PERSON_AREA * 2.2 and aspect >= 0.8:
            # Two people walking side by side merge into one blob.  Conservatively
            # count it as one rather than guessing a crowd size from geometry;
            # a real crowd separates into distinct blobs.
            return "person", 1
        if aspect >= 1.45:
            return "person", 1
        if aspect >= 0.55:
            return "object", 1
        return "vehicle", 1


class Tracker:
    """Greedy IoU/centroid tracker with velocity prediction."""

    def __init__(
        self,
        *,
        max_missing: int = 8,
        iou_weight: float = 0.7,
        distance_limit: float = 0.22,
    ) -> None:
        self.max_missing = max_missing
        self.iou_weight = iou_weight
        self.distance_limit = distance_limit
        self._tracks: list[Track] = []
        self._next_id = 1
        self._last_ts: float | None = None

    @property
    def tracks(self) -> list[Track]:
        return self._tracks

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._last_ts = None

    def update(self, detections: list[Detection], ts: float) -> list[Track]:
        dt = 1.0 / 10.0 if self._last_ts is None else max(1e-3, ts - self._last_ts)
        self._last_ts = ts

        candidates: list[tuple[float, int, int]] = []
        for ti, track in enumerate(self._tracks):
            predicted = track.predict(dt).clipped()
            for di, det in enumerate(detections):
                iou = predicted.iou(det.bbox)
                dist = predicted.center_distance(det.bbox)
                if iou <= 0.0 and dist > self.distance_limit:
                    continue
                proximity = max(0.0, 1.0 - dist / self.distance_limit)
                score = self.iou_weight * iou + (1.0 - self.iou_weight) * proximity
                if score > 0.08:
                    candidates.append((score, ti, di))

        candidates.sort(key=lambda item: item[0], reverse=True)
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for _score, ti, di in candidates:
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            self._tracks[ti].update(detections[di], ts)

        for ti, track in enumerate(self._tracks):
            if ti in matched_tracks:
                continue
            track.missing += 1
            track.bbox = track.predict(dt).clipped()
            track.velocity = (track.velocity[0] * 0.85, track.velocity[1] * 0.85)

        survivors: list[Track] = []
        for track in self._tracks:
            if track.missing > self.max_missing:
                continue
            bbox = track.bbox
            if bbox.x > 1.02 or bbox.y > 1.02 or bbox.x + bbox.w < -0.02:
                continue
            survivors.append(track)
        self._tracks = survivors

        for di, det in enumerate(detections):
            if di in matched_dets:
                continue
            self._tracks.append(
                Track(
                    id=self._next_id,
                    label=det.label,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    first_seen=ts,
                    last_seen=ts,
                    instances=det.instances,
                    history=[(ts, det.bbox.cx, det.bbox.cy)],
                    label_votes={det.label: 1},
                )
            )
            self._next_id += 1

        return self._tracks


def frame_energy(prev: np.ndarray | None, frame: np.ndarray) -> float:
    """Mean absolute frame difference, normalised to 0..1."""
    if prev is None or prev.shape != frame.shape:
        return 0.0
    diff = cv2.absdiff(frame, prev)
    return float(np.mean(diff)) / 255.0


def speed_of(track: Track) -> float:
    """Bounding-box-heights per second, kept finite for division-heavy features."""
    value = track.speed
    return value if math.isfinite(value) else 0.0
