from __future__ import annotations

from visioniq.core.store import EventStore
from visioniq.core.types import Event, Severity


def _event(event_id: int, kind: str = "anomaly", camera: str = "CAM-01") -> Event:
    return Event(
        id=event_id,
        ts=1_700_000_000.0 + event_id,
        camera_id=camera,
        camera_name="Main Entrance",
        zone="Ground / Zone A",
        kind=kind,
        severity=Severity.WARNING,
        message="test event",
        score=0.7,
        track_ids=(3, 4),
        detail={"persons": 2},
    )


def test_store_insert_recent_and_ack(tmp_path):
    store = EventStore(tmp_path / "events.db")
    first = store.insert(_event(1))
    second = store.insert(_event(2, "tailgate"))
    assert first != second

    rows = store.recent(10)
    assert [row.kind for row in rows] == ["tailgate", "anomaly"]
    assert rows[0].track_ids == (3, 4)
    assert rows[0].detail == {"persons": 2}
    assert store.acknowledge(first)
    assert not store.acknowledge(999)
    assert next(row for row in store.recent(10) if row.id == first).ack is True

    counts = store.counts(0.0)
    assert counts["warning"] == 2
    assert counts["total"] == 2
    store.close()
