from __future__ import annotations

import math

import numpy as np

from visionguard.benchmark import BenchmarkReport, _auroc, analyse, render_report
from visionguard.core.engine import Sample
from visionguard.core.types import CameraState, CameraView, MetricsView, Snapshot
from visionguard.tui.render import (
    banner_text,
    calibration_text,
    camera_header,
    event_summary,
    frame_to_text,
    rail_text,
    score_bar,
)
from visionguard.core.types import Event, Severity


def _sample(score: float, error: float, threshold: float, *, truth: tuple[str, ...] = (),
            flagged: bool = False, camera: str = "CAM-01") -> Sample:
    return Sample(camera, 1.0, score, error, threshold, flagged, truth, 0)


def test_auroc_perfect_and_inverted():
    assert _auroc(np.array([2.0, 3.0]), np.array([0.1, 0.2])) == 1.0
    assert _auroc(np.array([0.1, 0.2]), np.array([2.0, 3.0])) == 0.0
    assert math.isnan(_auroc(np.array([]), np.array([1.0])))


def test_ratio_normalisation_and_operating_point():
    samples = [
        _sample(0.9, 5.0, 1.0, truth=("running",), flagged=True),
        _sample(0.2, 0.2, 1.0, flagged=False),
        _sample(0.8, 4.0, 2.0, truth=("crowd",), flagged=True),
        _sample(0.1, 0.5, 2.0, flagged=False),
        _sample(0.55, 2.0, 1.0, truth=("running",), flagged=False),
        _sample(0.15, 1.0, 2.0, flagged=False),
    ]
    report = analyse(samples)
    assert report.windows == 6
    assert report.anomalous == 3
    assert report.normal == 3
    assert report.true_positive == 2
    assert report.false_positive == 0
    assert report.true_negative == 3
    assert report.false_negative == 1
    assert report.precision == 1.0
    assert report.recall == 2 / 3
    assert report.false_positive_rate == 0.0
    assert 0.0 <= report.auroc <= 1.0
    # The report exposes the operating-point sweep, including the live point.
    ratios = [row["ratio"] for row in report.operating_points]
    assert ratios == sorted(ratios)
    assert 1.0 in ratios
    live = next(row for row in report.operating_points if row["ratio"] == 1.0)
    assert 0.0 <= live["f1"] <= 1.0
    text = render_report(report)
    assert "VISIONGUARD SYNTHETIC BENCHMARK" in text
    assert "operating point sweep" in text
    assert "<- live" in text


def _view(camera_id: str = "CAM-01", state=CameraState.LIVE, score: float = 0.1) -> CameraView:
    return CameraView(
        id=camera_id,
        name="Main Entrance",
        zone="Ground / Zone A",
        state=state,
        score=score,
        threshold=1.0,
        raw_error=0.1,
        calibrating=False,
        progress=1.0,
        persons=1,
        vehicles=0,
        objects=0,
        fps=10.0,
        latency_ms=3.0,
        frame=None,
        tracks=(),
        last_event=None,
        backend="test",
        frames_seen=10,
    )


def test_frame_to_text_has_one_line_per_terminal_row():
    frame = np.zeros((24, 40), dtype=np.uint8)
    text = frame_to_text(frame, (), 20, 6)
    assert len(text.plain.split("\n")) == 6
    assert all(len(line) == 20 for line in text.plain.split("\n"))


def test_offline_and_calibration_render():
    assert "OFFLINE" in frame_to_text(None, (), 20, 3, offline=True).plain
    label = calibration_text(_view(state=CameraState.CALIBRATING), 30, 10)
    assert "CALIBRATING" in label.plain


def test_text_helpers_do_not_raise():
    view = _view(score=0.8)
    assert "CAM-01" in camera_header(view, 50).plain
    assert "CAM-01" in rail_text((view,), "CAM-01", 30).plain
    bar, colour = score_bar(0.9, 8)
    assert len(bar) == 8
    assert colour == "#ff3b3b"
    event = Event(
        id=1, ts=1_700_000_000.0, camera_id="CAM-01", camera_name="Main Entrance",
        zone="Ground / Zone A", kind="anomaly", severity=Severity.CRITICAL,
        message="unusual motion", score=0.9,
    )
    assert "unusual motion" in event_summary(event).plain
    assert "CRITICAL" in banner_text(_snapshot(event)).plain


def _snapshot(event: Event) -> Snapshot:
    metrics = MetricsView(0, 0, 0, 1, 1, 0, 1, 0, 1.0, 1.0, 1)
    return Snapshot(
        ts=1.0,
        clock="02:14:33",
        cameras=(_view(),),
        events=(event,),
        metrics=metrics,
        health={},
        paused=False,
        demo=True,
        model="test",
    )
