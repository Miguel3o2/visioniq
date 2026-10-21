"""Access-control correlation and compound security rules.

The autoencoder answers "is this scene statistically unusual?".  This module
answers "does it matter?" by correlating detections with access-control (ACAM)
badge events and site policy: tailgating, unbadged access, after-hours motion,
loitering, crowding, abandoned objects and vehicles in restricted areas.

Every rule has an explicit cooldown and an evidence payload, so the operator
can see *why* an alarm fired instead of trusting an opaque score.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from visioniq.config import CameraConfig, RuleConfig
from visioniq.core.types import (
    AnomalyResult,
    BadgeEvent,
    Event,
    Severity,
    Track,
)
from visioniq.vision.features import CameraFeatures


@dataclass(slots=True)
class Entry:
    ts: float
    track_id: int
    matched: bool = False
    resolved: bool = False


@dataclass(slots=True)
class BadgeSlot:
    event: BadgeEvent
    used: bool = False


@dataclass(slots=True)
class CorrelatorStats:
    entries: int = 0
    badges: int = 0
    matched: int = 0
    suppressed: int = 0


class CorrelationEngine:
    """Per-camera rule evaluation."""

    def __init__(self, camera: CameraConfig, rules: RuleConfig) -> None:
        self.camera = camera
        self.rules = rules
        self._badges: deque[BadgeSlot] = deque(maxlen=64)
        self._entries: deque[Entry] = deque(maxlen=64)
        self._seen_entries: set[int] = set()
        self._cooldowns: dict[tuple[str, str], float] = {}
        self._was_anomaly = False
        self._loitered: set[int] = set()
        self._abandoned: set[int] = set()
        self._last_cleanup = 0.0
        self.stats = CorrelatorStats()

    # -- helpers -----------------------------------------------------------
    def _cooled(self, kind: str, subject: str, ts: float, cooldown: float) -> bool:
        key = (kind, subject)
        last = self._cooldowns.get(key)
        if last is not None and ts - last < cooldown:
            self.stats.suppressed += 1
            return False
        self._cooldowns[key] = ts
        return True

    def _event(self, ts: float, kind: str, severity: Severity, message: str,
               *, score: float = 0.0, track_ids: tuple[int, ...] = (),
               detail: dict | None = None) -> Event:
        return Event(
            id=0,
            ts=ts,
            camera_id=self.camera.id,
            camera_name=self.camera.name,
            zone=self.camera.zone,
            kind=kind,
            severity=severity,
            message=message,
            score=score,
            track_ids=track_ids,
            detail=detail or {},
        )

    @staticmethod
    def _hour(ts: float) -> float:
        lt = time.localtime(ts)
        return lt.tm_hour + lt.tm_min / 60.0 + lt.tm_sec / 3600.0

    # -- main --------------------------------------------------------------
    def observe(
        self,
        ts: float,
        tracks: list[Track],
        features: CameraFeatures,
        anomaly: AnomalyResult,
        badges: list[BadgeEvent],
    ) -> list[Event]:
        events: list[Event] = []
        active = [t for t in tracks if t.confirmed and t.missing == 0]
        people = [t for t in active if t.label == "person"]
        vehicles = [t for t in active if t.label == "vehicle"]
        objects = [t for t in active if t.label == "object"]

        for badge in badges:
            self._badges.append(BadgeSlot(badge))
            self.stats.badges += 1
            self._match_late_badge(badge, ts)

        events.extend(self._entries_and_tailgate(ts, people))
        events.extend(self._access_rules(ts, people))

        if self.rules.has("anomaly") and anomaly.is_anomaly and not anomaly.calibrating:
            if not self._was_anomaly and self._cooled("anomaly", "camera", ts, self.rules.anomaly_cooldown):
                top = ", ".join(f"{name} ({share:.0%})" for name, share in anomaly.top_features)
                severity = Severity.CRITICAL if anomaly.score >= 0.9 else Severity.WARNING
                events.append(
                    self._event(
                        ts, "anomaly", severity,
                        f"Sequence anomaly {anomaly.score:.2f} - unusual motion pattern"
                        + (f" [{top}]" if top else ""),
                        score=anomaly.score,
                        detail={
                            "raw_error": round(anomaly.raw_error, 4),
                            "threshold": round(anomaly.threshold, 4),
                            "features": [name for name, _ in anomaly.top_features],
                        },
                    )
                )
        self._was_anomaly = anomaly.is_anomaly and not anomaly.calibrating

        if self.rules.has("crowding") and features.persons > self.camera.capacity:
            if self._cooled("crowding", "camera", ts, 30.0):
                events.append(
                    self._event(
                        ts, "crowding", Severity.WARNING,
                        f"Occupancy {features.persons} exceeds capacity {self.camera.capacity}",
                        score=float(features.persons),
                        detail={"persons": features.persons, "capacity": self.camera.capacity},
                    )
                )

        if self.rules.has("loitering") and self.camera.restricted:
            for track in people:
                if track.id in self._loitered or track.dwell < self.camera.loiter_seconds:
                    continue
                self._loitered.add(track.id)
                if self._cooled("loitering", f"track:{track.id}", ts, self.camera.loiter_seconds):
                    events.append(
                        self._event(
                            ts, "loitering", Severity.WARNING,
                            f"Loitering {track.dwell:.0f}s in {self.camera.zone}",
                            score=min(1.0, track.dwell / (self.camera.loiter_seconds * 2)),
                            track_ids=(track.id,),
                            detail={"dwell_s": round(track.dwell, 1),
                                    "required_s": self.camera.loiter_seconds},
                        )
                    )

        if self.rules.has("abandoned"):
            for obj in objects:
                if obj.id in self._abandoned:
                    continue
                if obj.dwell >= self.rules.abandoned_seconds and obj.speed < 0.3:
                    self._abandoned.add(obj.id)
                    if self._cooled("abandoned", f"track:{obj.id}", ts, 300.0):
                        events.append(
                            self._event(
                                ts, "abandoned", Severity.WARNING,
                                f"Stationary object for {obj.dwell:.0f}s in {self.camera.zone}",
                                score=min(1.0, obj.dwell / (self.rules.abandoned_seconds * 2)),
                                track_ids=(obj.id,),
                                detail={"dwell_s": round(obj.dwell, 1),
                                        "size": [round(obj.bbox.w, 3), round(obj.bbox.h, 3)]},
                            )
                        )

        if self.rules.has("after_hours") and self.camera.restricted and people:
            hour = self._hour(ts)
            inside = [track for track in people if self._in_restricted(track)]
            if inside and self.camera.is_after_hours(hour) and self._cooled("after_hours", "camera", ts, 240.0):
                minute = int((hour % 1.0) * 60)
                events.append(
                    self._event(
                        ts, "after_hours", Severity.WARNING,
                        f"After-hours motion in {self.camera.zone} ({int(hour):02d}:{minute:02d})",
                        score=min(1.0, len(inside) / max(1, self.camera.capacity)),
                        track_ids=tuple(t.id for t in inside[:4]),
                        detail={"persons": len(inside), "hour": round(hour, 2)},
                    )
                )

        if self.rules.has("vehicle_restricted") and self.camera.restricted:
            for vehicle in vehicles:
                if self._cooled("vehicle_restricted", f"track:{vehicle.id}", ts, 120.0):
                    events.append(
                        self._event(
                            ts, "vehicle_restricted", Severity.CRITICAL,
                            f"Vehicle detected in restricted area {self.camera.zone}",
                            score=0.9,
                            track_ids=(vehicle.id,),
                            detail={"speed": round(vehicle.speed, 2)},
                        )
                    )

        if ts - self._last_cleanup > 60.0:
            self._cleanup(ts)
        return events

    # -- access control ----------------------------------------------------
    def _unused_badge_near(self, ts: float, window: float) -> BadgeSlot | None:
        best: BadgeSlot | None = None
        best_delta = window + 1.0
        for slot in self._badges:
            if slot.used or not slot.event.granted:
                continue
            delta = abs(slot.event.ts - ts)
            if delta <= window and delta < best_delta:
                best, best_delta = slot, delta
        return best

    def _match_late_badge(self, badge: BadgeEvent, ts: float) -> None:
        """Attach a badge that arrived after the visual entry."""
        if not badge.granted:
            return
        for entry in self._entries:
            if entry.matched or entry.resolved:
                continue
            if abs(entry.ts - badge.ts) <= self.rules.badge_window:
                entry.matched = True
                self.stats.matched += 1
                slot = self._badges[-1]
                slot.used = True
                return

    def _entries_and_tailgate(self, ts: float, people: list[Track]) -> list[Event]:
        """Record door crossings and match them against badge events."""
        zone = self.camera.entry_zone
        if zone is None:
            return []
        for track in people:
            if track.id in self._seen_entries:
                continue
            if not zone.contains(track.bbox.cx, track.bbox.cy) or track.dwell < 0.4:
                continue
            self._seen_entries.add(track.id)
            entry = Entry(ts=ts, track_id=track.id)
            self._entries.append(entry)
            self.stats.entries += 1

            slot = self._unused_badge_near(ts, self.rules.badge_window)
            if slot is not None:
                slot.used = True
                entry.matched = True
                self.stats.matched += 1
        return []

    def _in_restricted(self, track: Track | None) -> bool:
        if track is None:
            return False
        zones = self.camera.restricted_zones
        if not zones:
            return self.camera.restricted
        return any(zone.rect.contains(track.bbox.cx, track.bbox.cy) for zone in zones)

    def _access_rules(self, ts: float, people: list[Track]) -> list[Event]:
        """Tailgating and badge-less access.

        A composite alarm needs a *tight* cluster of entries: two people
        crossing a door within a few seconds where at least one has no badge.
        Two unrelated crossings minutes apart are not a tailgate, which is why
        the check is deferred until the badge correlation window has settled.
        """
        if not self.camera.access_controlled:
            return []
        events: list[Event] = []
        by_id = {track.id: track for track in people}
        grace = min(2.5, self.rules.badge_window)

        for entry in self._entries:
            if entry.resolved or entry.matched:
                continue
            if ts - entry.ts < grace:
                continue
            slot = self._unused_badge_near(entry.ts, self.rules.badge_window)
            if slot is not None:
                slot.used = True
                entry.matched = True
                entry.resolved = True
                self.stats.matched += 1
                continue
            track = by_id.get(entry.track_id)
            # Only report once the person is actually inside a restricted area;
            # the doorway itself is a public space.
            if self.rules.has("unbadged_access") and self._in_restricted(track):
                entry.resolved = True
                if self._cooled("unbadged_access", f"track:{entry.track_id}", ts, 120.0):
                    events.append(
                        self._event(
                            ts, "unbadged_access", Severity.CRITICAL,
                            f"Motion without badge in restricted area {self.camera.zone}",
                            score=0.85,
                            track_ids=(entry.track_id,),
                            detail={"entry_ts": round(entry.ts, 2),
                                    "badge_window_s": self.rules.badge_window,
                                    "zone": self.camera.zone},
                        )
                    )
            elif ts - entry.ts >= self.rules.badge_window:
                entry.resolved = True

        if self.rules.has("tailgate"):
            cluster_window = min(self.rules.tailgate_window, 3.0)
            recent = [e for e in self._entries if ts - e.ts <= cluster_window]
            if len(recent) >= 2 and ts - recent[-1].ts >= grace:
                unmatched = [e for e in recent if not e.matched]
                if unmatched and self._cooled("tailgate", "camera", ts, 20.0):
                    events.append(
                        self._event(
                            ts, "tailgate", Severity.CRITICAL,
                            f"Tailgate: {len(recent)} entries, "
                            f"{len(recent) - len(unmatched)} badge(s) within "
                            f"{cluster_window:g}s",
                            score=0.95,
                            track_ids=tuple(e.track_id for e in recent),
                            detail={
                                "entries": len(recent),
                                "unmatched": len(unmatched),
                                "window_s": cluster_window,
                            },
                        )
                    )
        return events

    def _cleanup(self, ts: float) -> None:
        self._last_cleanup = ts
        horizon = ts - max(600.0, self.rules.badge_window * 4)
        self._cooldowns = {k: v for k, v in self._cooldowns.items() if v >= horizon}
        self._badges = deque((b for b in self._badges if b.event.ts >= horizon), maxlen=64)
        self._entries = deque((e for e in self._entries if e.ts >= horizon), maxlen=64)
        # Track ids are never reused, but bound these dedupe sets over long runs.
        if len(self._loitered) > 5000:
            self._loitered.clear()
        if len(self._abandoned) > 5000:
            self._abandoned.clear()
