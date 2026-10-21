"""SQLite persistence for events.

Events are written on the pipeline thread and read from the console thread, so
the connection is created with ``check_same_thread=False`` and guarded by a
lock.  WAL mode keeps reads cheap while writes are in flight.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from visioniq.core.types import Event, Severity

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,
    camera_id   TEXT    NOT NULL,
    camera_name TEXT    NOT NULL,
    zone        TEXT    NOT NULL,
    kind        TEXT    NOT NULL,
    severity    TEXT    NOT NULL,
    message     TEXT    NOT NULL,
    score       REAL    NOT NULL DEFAULT 0,
    track_ids   TEXT    NOT NULL DEFAULT '[]',
    detail      TEXT    NOT NULL DEFAULT '{}',
    ack         INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events (ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_severity ON events (severity, ts DESC);
"""


class EventStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def insert(self, event: Event) -> int:
        with self._lock:
            cursor = self._conn.execute(
                """INSERT INTO events
                   (ts, camera_id, camera_name, zone, kind, severity, message, score, track_ids, detail, ack)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.ts,
                    event.camera_id,
                    event.camera_name,
                    event.zone,
                    event.kind,
                    event.severity.value,
                    event.message,
                    event.score,
                    json.dumps(list(event.track_ids)),
                    json.dumps(event.detail),
                    int(event.ack),
                ),
            )
            self._conn.commit()
            return int(cursor.lastrowid)

    def acknowledge(self, event_id: int) -> bool:
        with self._lock:
            cursor = self._conn.execute("UPDATE events SET ack = 1 WHERE id = ?", (event_id,))
            self._conn.commit()
            return cursor.rowcount > 0

    def recent(self, limit: int = 100) -> list[Event]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT id, ts, camera_id, camera_name, zone, kind, severity, message,
                          score, track_ids, detail, ack
                   FROM events ORDER BY ts DESC, id DESC LIMIT ?""",
                (max(1, limit),),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def counts(self, since_ts: float) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT severity, COUNT(*) FROM events WHERE ts >= ? GROUP BY severity",
                (since_ts,),
            ).fetchall()
            total = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        result = {"critical": 0, "warning": 0, "info": 0, "total": int(total)}
        for severity, count in rows:
            result[str(severity)] = int(count)
        return result

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _row_to_event(row: tuple) -> Event:
        (event_id, ts, camera_id, camera_name, zone, kind, severity,
         message, score, track_ids, detail, ack) = row
        try:
            tracks = tuple(int(v) for v in json.loads(track_ids))
        except (TypeError, ValueError):
            tracks = ()
        try:
            detail_dict = json.loads(detail)
        except (TypeError, ValueError):
            detail_dict = {}
        return Event(
            id=int(event_id),
            ts=float(ts),
            camera_id=str(camera_id),
            camera_name=str(camera_name),
            zone=str(zone),
            kind=str(kind),
            severity=Severity.parse(str(severity)),
            message=str(message),
            score=float(score),
            track_ids=tracks,
            detail=detail_dict if isinstance(detail_dict, dict) else {},
            ack=bool(ack),
        )
