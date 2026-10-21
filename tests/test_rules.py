from __future__ import annotations

import time
from datetime import datetime

import numpy as np

from visionguard.config import CameraConfig, Rect, RuleConfig, ZoneConfig
from visionguard.core.types import AnomalyResult, BadgeEvent, BBox, Track
from visionguard.correlation.rules import CorrelationEngine
from visionguard.vision.features import CameraFeatures


def _camera(**overrides) -> CameraConfig:
    values = dict(
        id="CAM-01",
        name="Test Gate",
        zone="Ground / Test",
        source="synthetic:lobby",
        restricted=True,
        access_controlled=True,
        capacity=3,
        loiter_seconds=90.0,
        seed=1,
        zones=(
            ZoneConfig("door", Rect.parse([0.70, 0.30, 0.22, 0.50]), "entry"),
            ZoneConfig("secure", Rect.parse([0.02, 0.55, 0.55, 0.35]), "restricted"),
        ),
    )
    values.update(overrides)
    return CameraConfig(**values)


def _track(track_id: int, label: str, cx: float, cy: float, *,
           first_seen: float = 0.0, last_seen: float = 1.0,
           vx: float = 0.0) -> Track:
    bbox = BBox(cx - 0.03, cy - 0.1, 0.06, 0.20)
    track = Track(
        id=track_id,
        label=label,
        bbox=bbox,
        confidence=0.9,
        first_seen=first_seen,
        last_seen=last_seen,
        velocity=(vx, 0.0),
        confirmed=True,
    )
    track.history = [(first_seen, cx - vx, cy), (last_seen, cx, cy)]
    return track


def _features(persons: int = 0, vehicles: int = 0, objects: int = 0) -> CameraFeatures:
    return CameraFeatures(np.zeros(12, dtype=np.float32), persons, vehicles, objects)


def _normal() -> AnomalyResult:
    return AnomalyResult(score=0.1, raw_error=0.1, threshold=1.0,
                         is_anomaly=False, calibrating=False)


def test_tailgate_requires_a_tight_pair_with_one_badge():
    camera = _camera()
    engine = CorrelationEngine(camera, RuleConfig())
    first = _track(1, "person", 0.84, 0.50)
    second = _track(2, "person", 0.88, 0.55)
    events = engine.observe(1.0, [first, second], _features(2), _normal(),
                            [BadgeEvent(ts=1.0, person="P1", door="lobby", granted=True)])
    assert not [e for e in events if e.kind == "tailgate"]
    events = engine.observe(4.0, [second], _features(1), _normal(), [])
    tailgates = [e for e in events if e.kind == "tailgate"]
    assert len(tailgates) == 1
    assert tailgates[0].severity.value == "critical"
    assert tailgates[0].detail["entries"] == 2
    assert tailgates[0].detail["unmatched"] == 1


def test_unbadged_access_fires_once_the_person_reaches_the_secure_zone():
    camera = _camera()
    engine = CorrelationEngine(camera, RuleConfig())
    person = _track(5, "person", 0.85, 0.50)
    engine.observe(1.0, [person], _features(1), _normal(), [])
    person.bbox = BBox(0.20, 0.65, 0.06, 0.20)
    events = engine.observe(4.0, [person], _features(1), _normal(), [])
    unbadged = [e for e in events if e.kind == "unbadged_access"]
    assert len(unbadged) == 1
    assert unbadged[0].severity.value == "critical"


def test_matching_badge_suppresses_unbadged_access():
    camera = _camera()
    engine = CorrelationEngine(camera, RuleConfig())
    person = _track(6, "person", 0.85, 0.50)
    engine.observe(1.0, [person], _features(1), _normal(),
                   [BadgeEvent(ts=1.0, person="P6", door="lobby", granted=True)])
    person.bbox = BBox(0.20, 0.65, 0.06, 0.20)
    events = engine.observe(4.0, [person], _features(1), _normal(), [])
    assert not [e for e in events if e.kind in {"unbadged_access", "tailgate"}]


def test_after_hours_requires_restricted_zone_and_night_hour():
    camera = _camera(access_controlled=False)
    engine = CorrelationEngine(camera, RuleConfig())
    ts = datetime.now().replace(hour=2, minute=14, second=0, microsecond=0).timestamp()
    outside = _track(7, "person", 0.85, 0.45)
    assert not [e for e in engine.observe(ts, [outside], _features(1), _normal(), [])
                if e.kind == "after_hours"]

    inside = _track(8, "person", 0.20, 0.65)
    events = engine.observe(ts + 1.0, [inside], _features(1), _normal(), [])
    after = [e for e in events if e.kind == "after_hours"]
    assert len(after) == 1
    assert "02:14" in after[0].message


def test_crowding_uses_strictly_greater_than_capacity():
    camera = _camera(restricted=False, access_controlled=False, capacity=2, zones=())
    engine = CorrelationEngine(camera, RuleConfig())
    at_capacity = engine.observe(1.0, [], _features(2), _normal(), [])
    assert not [e for e in at_capacity if e.kind == "crowding"]
    over = engine.observe(31.0, [], _features(3), _normal(), [])
    crowding = [e for e in over if e.kind == "crowding"]
    assert len(crowding) == 1
    assert "3 exceeds capacity 2" in crowding[0].message


def test_loitering_and_abandoned_rules():
    camera = _camera(access_controlled=False)
    engine = CorrelationEngine(camera, RuleConfig())
    loiterer = _track(9, "person", 0.20, 0.65, first_seen=0.0, last_seen=95.0, vx=0.0)
    events = engine.observe(95.0, [loiterer], _features(1), _normal(), [])
    assert [e for e in events if e.kind == "loitering"]

    bag = _track(10, "object", 0.45, 0.70, first_seen=0.0, last_seen=30.0, vx=0.0)
    events = engine.observe(200.0, [bag], _features(objects=1), _normal(), [])
    assert [e for e in events if e.kind == "abandoned"]


def test_vehicle_in_restricted_area_and_anomaly_cooldown():
    camera = _camera(zones=(), access_controlled=False)
    engine = CorrelationEngine(camera, RuleConfig())
    car = _track(11, "vehicle", 0.5, 0.6)
    events = engine.observe(1.0, [car], _features(vehicles=1), _normal(), [])
    assert [e for e in events if e.kind == "vehicle_restricted"]

    anomaly = AnomalyResult(score=0.95, raw_error=3.0, threshold=1.0,
                            is_anomaly=True, calibrating=False,
                            top_features=(("peak speed", 0.8),))
    first = engine.observe(5.0, [], _features(), anomaly, [])
    second = engine.observe(6.0, [], _features(), anomaly, [])
    assert [e for e in first if e.kind == "anomaly"]
    assert not [e for e in second if e.kind == "anomaly"]
