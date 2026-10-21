from __future__ import annotations

import numpy as np

from visionguard.core.types import Detection
from visionguard.vision.detector import DetectorConfig, MotionDetector, Tracker, frame_energy
from visionguard.vision.features import FeatureExtractor
from visionguard.vision.source import SyntheticSource


def _frames(source: SyntheticSource, count: int) -> list[np.ndarray]:
    return [source.read() for _ in range(count)]


def test_detector_finds_moving_person_and_tracker_keeps_identity():
    source = SyntheticSource("lobby", seed=3, traffic=0.0)
    source.inject("running", duration=3.0)
    frames = _frames(source, 24)

    detector = MotionDetector(DetectorConfig(learning_rate=0.05))
    tracker = Tracker()
    assert detector.detect(frames[0]) == []  # first frame establishes background

    ids: set[int] = set()
    for index, frame in enumerate(frames[1:], start=1):
        detections = detector.detect(frame, protect=[t.bbox for t in tracker.tracks])
        tracks = tracker.update(detections, index / 10.0)
        ids.update(t.id for t in tracks if t.confirmed)
    assert ids, "the moving actor should be tracked"
    assert max(t.dwell for t in tracks) > 0.2


def test_background_is_frozen_inside_protected_boxes():
    """A stationary person must not be absorbed into the background."""

    source = SyntheticSource("server", seed=7, traffic=0.0)
    source.inject("loitering", duration=20.0)
    frames = _frames(source, 80)

    detector = MotionDetector(DetectorConfig(learning_rate=0.10))
    tracker = Tracker()
    detected_after = 0
    for index, frame in enumerate(frames):
        detections = detector.detect(frame, protect=[t.bbox for t in tracker.tracks])
        tracker.update(detections, index / 10.0)
        if index > 60 and detections:
            detected_after += 1
    assert detected_after > 0, "protection must keep a stationary actor in the foreground"


def test_tracker_ignores_unmatched_detection_after_timeout():
    tracker = Tracker(max_missing=2)
    det = Detection(bbox=_box(0.2, 0.2), label="person", confidence=0.9)
    tracker.update([det], 0.0)
    tracker.update([det], 0.1)
    assert len(tracker.tracks) == 1
    for step in range(4):
        tracker.update([], 0.2 + step * 0.1)
    assert tracker.tracks == []


def test_features_count_and_window():
    extractor = FeatureExtractor(new_track_window=3.0)
    tracks = [
        _track(1, "person", 0.2, 0.5, first_seen=0.0, last_seen=2.0, vx=0.1),
        _track(2, "person", 0.5, 0.4, first_seen=0.0, last_seen=2.0, vx=-0.1),
        _track(3, "vehicle", 0.6, 0.6, first_seen=0.0, last_seen=2.0, vx=0.0),
    ]
    features = extractor.extract(tracks, motion_energy=0.1, ts=2.1)
    assert features.persons == 2
    assert features.vehicles == 1
    assert features.objects == 0
    assert features.values.shape == (12,)
    assert features.values[0] == 2 / 8.0
    assert features.values[4] == 0.1 * 25.0


def test_frame_energy_is_zero_for_first_frame():
    frame = np.zeros((20, 30), dtype=np.uint8)
    assert frame_energy(None, frame) == 0.0
    assert frame_energy(frame, frame) == 0.0
    assert frame_energy(frame, np.full_like(frame, 255)) == 1.0


def _box(cx: float, cy: float):
    from visionguard.core.types import BBox

    return BBox(cx, cy, 0.06, 0.22)


def _track(track_id: int, label: str, cx: float, cy: float, *, first_seen: float,
           last_seen: float, vx: float = 0.0):
    from visionguard.core.types import BBox, Track

    track = Track(
        id=track_id,
        label=label,
        bbox=BBox(cx, cy, 0.06, 0.22),
        confidence=0.9,
        first_seen=first_seen,
        last_seen=last_seen,
        velocity=(vx, 0.0),
        confirmed=True,
    )
    track.history = [(first_seen, cx - vx, cy), (last_seen, cx, cy)]
    return track
