"""Per-camera feature extraction.

The autoencoder never sees pixels: it sees a short sequence of interpretable
scene statistics.  That keeps the model small enough to train on-device in a
few seconds, makes anomalies explainable ("person count", "peak speed"), and
keeps the console useful even while the model is calibrating.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from visionguard.core.types import FEATURE_NAMES, Track, TrackView


@dataclass(slots=True, frozen=True)
class CameraFeatures:
    values: np.ndarray
    persons: int
    vehicles: int
    objects: int


def _clip(value: float, low: float = 0.0, high: float = 6.0) -> float:
    if not math.isfinite(value):
        return low
    return float(min(high, max(low, value)))


class FeatureExtractor:
    def __init__(self, new_track_window: float = 3.0) -> None:
        self.new_track_window = new_track_window

    def extract(self, tracks: list[Track], motion_energy: float, ts: float) -> CameraFeatures:
        confirmed = [t for t in tracks if t.confirmed and t.missing == 0]
        persons = [t for t in confirmed if t.label == "person"]
        vehicles = [t for t in confirmed if t.label == "vehicle"]
        objects = [t for t in confirmed if t.label == "object"]

        person_count = sum(max(1, t.instances) for t in persons)
        vehicle_count = sum(max(1, t.instances) for t in vehicles)
        object_count = sum(max(1, t.instances) for t in objects)

        occupancy = sum(t.bbox.area for t in persons)
        speeds = [_clip(t.speed, 0.0, 8.0) for t in persons]
        mean_speed = float(np.mean(speeds)) if speeds else 0.0
        max_speed = float(np.max(speeds)) if speeds else 0.0
        speed_spread = float(np.std(speeds)) if len(speeds) > 1 else 0.0
        new_tracks = sum(1 for t in confirmed if ts - t.first_seen <= self.new_track_window)
        dwell = max((t.dwell for t in persons), default=0.0)
        centers = np.array([[t.bbox.cx, t.bbox.cy] for t in persons], dtype=np.float32)
        dispersion = float(np.std(centers, axis=0).mean()) if len(centers) > 1 else 0.0

        flow = 0.0
        moving = [t for t in persons if t.speed > 0.05]
        if moving:
            vx = float(np.mean([t.velocity[0] for t in moving]))
            vy = float(np.mean([t.velocity[1] for t in moving]))
            mean_mag = float(np.mean([math.hypot(*t.velocity) for t in moving])) + 1e-6
            flow = min(1.0, math.hypot(vx, vy) / mean_mag)

        values = np.array(
            [
                _clip(person_count / 8.0, 0.0, 3.0),
                _clip(vehicle_count / 4.0, 0.0, 3.0),
                _clip(object_count / 4.0, 0.0, 3.0),
                _clip(occupancy * 12.0, 0.0, 4.0),
                _clip(motion_energy * 25.0, 0.0, 4.0),
                _clip(mean_speed * 0.8, 0.0, 4.0),
                _clip(max_speed * 0.8, 0.0, 5.0),
                _clip(speed_spread * 0.8, 0.0, 4.0),
                _clip(new_tracks / 4.0, 0.0, 3.0),
                _clip(dwell / 45.0, 0.0, 4.0),
                _clip(dispersion * 4.0, 0.0, 4.0),
                _clip(flow, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        assert values.shape == (len(FEATURE_NAMES),)
        return CameraFeatures(values, person_count, vehicle_count, object_count)


def track_views(tracks: list[Track], limit: int = 24) -> tuple[TrackView, ...]:
    views = [
        TrackView(
            id=t.id,
            label=t.label,
            bbox=t.bbox,
            speed=t.speed,
            dwell=t.dwell,
            instances=t.instances,
            confirmed=t.confirmed,
        )
        for t in tracks
        if t.confirmed and t.missing == 0
    ]
    views.sort(key=lambda v: v.bbox.area, reverse=True)
    return tuple(views[:limit])
