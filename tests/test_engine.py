from __future__ import annotations

import time

from visioniq.config import AppConfig, CameraConfig, ModelConfig
from visioniq.core.engine import Engine
from visioniq.vision.source import DemoScript


def _config() -> AppConfig:
    camera = CameraConfig(
        id="CAM-T",
        name="Test Scene",
        zone="Lab / Zone T",
        source="synthetic:server",
        restricted=False,
        capacity=2,
        seed=5,
        fps=10.0,
        traffic=0.5,
    )
    model = ModelConfig(
        backend="numpy",
        window=8,
        stride=1,
        hidden=8,
        warmup_windows=16,
        epochs=1,
        batch_size=8,
        threshold_percentile=90.0,
        threshold_multiplier=1.0,
        error_floor=0.01,
    )
    return AppConfig(cameras=[camera], model=model).validate()


def test_engine_runs_calibrates_and_persists(tmp_path):
    config = _config()
    script = DemoScript(steps=[(0.5, "crowd", "CAM-T")], duration=3.0, anomaly_seconds=2.0)
    engine = Engine(config, demo_script=script, speed=2.0, preroll=True,
                    record=True, root=tmp_path)
    engine.start()
    try:
        assert engine.wait_ready(60.0), engine.status()
        snapshot = engine.snapshot()
        assert snapshot is not None
        assert snapshot.cameras[0].id == "CAM-T"
        assert not snapshot.cameras[0].calibrating
        assert snapshot.cameras[0].frames_seen > 0

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and engine.metrics().frames < 10:
            time.sleep(0.05)
        assert engine.metrics().frames >= 10

        ok, message = engine.inject("crowd", "CAM-T")
        assert ok, message
        path = engine.save_snapshot("CAM-T")
        assert path is not None and path.exists()

        engine.set_paused(True)
        time.sleep(0.1)
        assert engine.paused
        engine.set_paused(False)

        samples = engine.recorded_samples()
        assert samples and all(sample.camera_id == "CAM-T" for sample in samples)
        assert engine.metrics().windows_scored > 0
    finally:
        engine.stop()

    db = tmp_path / "data" / "visioniq.db"
    assert db.exists()
    from visioniq.core.store import EventStore

    store = EventStore(db)
    rows = store.recent(10)
    store.close()
    assert rows, "the scripted anomaly should produce at least one rule event"


def test_engine_acknowledges_events(tmp_path):
    config = _config()
    script = DemoScript(steps=[(0.2, "crowd", "CAM-T")], duration=2.0, anomaly_seconds=2.0)
    engine = Engine(config, demo_script=script, speed=2.0, preroll=True, root=tmp_path)
    engine.start()
    try:
        assert engine.wait_ready(60.0)
        deadline = time.monotonic() + 6.0
        event_id = None
        while time.monotonic() < deadline and event_id is None:
            snapshot = engine.snapshot()
            if snapshot and snapshot.events:
                event_id = snapshot.events[0].id
            time.sleep(0.05)
        assert event_id is not None, "expected at least one event"
        assert engine.acknowledge(event_id)
        snapshot = engine.snapshot()
        assert next(e for e in snapshot.events if e.id == event_id).ack is True
    finally:
        engine.stop()
