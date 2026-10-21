"""Measured performance of the anomaly detector.

The original project shipped a hard-coded ``AUROC 0.947``.  This module
replaces it with numbers computed from an actual run:

* synthetic scenes are used so the ground truth is known exactly;
* a schedule of isolated injections creates clean normal and anomalous
  windows per camera;
* the report states precision / recall / FPR at the *calibrated* operating
  point (the hysteresis flag), the best achievable F1, and AUROC over the
  normalised reconstruction error.

Normalising each window's error by that camera's own threshold is important:
raw errors are not comparable between cameras (a static server room can have a
threshold two orders of magnitude lower than a busy lobby).

It is a synthetic validation, not a claim about real-world accuracy.  That is
stated in the output as well as in the README.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from visionguard.core.engine import Engine, Sample


@dataclass(slots=True)
class BenchmarkReport:
    windows: int = 0
    anomalous: int = 0
    normal: int = 0
    auroc: float = float("nan")
    # Operating point: the hysteresis flag produced by the live scorer.
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    specificity: float = 0.0
    false_positive_rate: float = 0.0
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0
    # Best F1 attainable by moving the error/threshold ratio.
    best_f1: float = 0.0
    best_ratio: float = 0.0
    mean_ratio_anomalous: float = 0.0
    mean_ratio_normal: float = 0.0
    p95_ratio_normal: float = 0.0
    mean_score_anomalous: float = 0.0
    mean_score_normal: float = 0.0
    per_camera: dict[str, dict[str, float]] = field(default_factory=dict)
    per_kind: dict[str, dict[str, float]] = field(default_factory=dict)
    # Precision/recall at a range of error/threshold ratios.  The live
    # operating point is one row (ratio 1.00); the table makes the
    # sensitivity/false-alarm trade-off visible instead of hiding it behind a
    # single headline number.
    operating_points: list[dict[str, float]] = field(default_factory=list)
    event_counts: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    fps: float = 0.0
    duration_s: float = 0.0
    caveat: str = "Synthetic validation only; ground truth comes from the scene injector."

    def as_dict(self) -> dict:
        return {
            "windows": self.windows,
            "anomalous": self.anomalous,
            "normal": self.normal,
            "auroc": self.auroc,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "specificity": self.specificity,
            "false_positive_rate": self.false_positive_rate,
            "confusion": {
                "tp": self.true_positive,
                "fp": self.false_positive,
                "tn": self.true_negative,
                "fn": self.false_negative,
            },
            "best_f1": self.best_f1,
            "best_ratio": self.best_ratio,
            "mean_ratio_anomalous": self.mean_ratio_anomalous,
            "mean_ratio_normal": self.mean_ratio_normal,
            "p95_ratio_normal": self.p95_ratio_normal,
            "mean_score_anomalous": self.mean_score_anomalous,
            "mean_score_normal": self.mean_score_normal,
            "per_camera": self.per_camera,
            "per_kind": self.per_kind,
            "operating_points": self.operating_points,
            "event_counts": self.event_counts,
            "latency_ms": self.latency_ms,
            "fps": self.fps,
            "duration_s": self.duration_s,
            "caveat": self.caveat,
        }


def ratio_of(sample: Sample) -> float:
    """Reconstruction error normalised by this camera's calibrated threshold."""

    return sample.error / max(sample.threshold, 1e-9)


def _auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    scores = np.concatenate([positive, negative])
    labels = np.concatenate([np.ones(len(positive)), np.zeros(len(negative))])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = average_rank
        i = j + 1
    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _confusion(samples: list[Sample], predict) -> tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for sample in samples:
        flagged = predict(sample)
        anomalous = bool(sample.ground_truth)
        if flagged and anomalous:
            tp += 1
        elif flagged and not anomalous:
            fp += 1
        elif not flagged and anomalous:
            fn += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def _scores(tp: int, fp: int, tn: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def analyse(samples: list[Sample], engine: Engine | None = None) -> BenchmarkReport:
    samples = [s for s in samples if math.isfinite(s.score) and math.isfinite(s.error)]
    report = BenchmarkReport()
    if not samples:
        return report

    pos_ratio = np.array([ratio_of(s) for s in samples if s.ground_truth], dtype=float)
    neg_ratio = np.array([ratio_of(s) for s in samples if not s.ground_truth], dtype=float)
    pos_score = np.array([s.score for s in samples if s.ground_truth], dtype=float)
    neg_score = np.array([s.score for s in samples if not s.ground_truth], dtype=float)

    report.windows = len(samples)
    report.anomalous = len(pos_ratio)
    report.normal = len(neg_ratio)
    report.auroc = _auroc(pos_ratio, neg_ratio)
    report.mean_ratio_anomalous = float(pos_ratio.mean()) if len(pos_ratio) else 0.0
    report.mean_ratio_normal = float(neg_ratio.mean()) if len(neg_ratio) else 0.0
    report.p95_ratio_normal = float(np.percentile(neg_ratio, 95)) if len(neg_ratio) else 0.0
    report.mean_score_anomalous = float(pos_score.mean()) if len(pos_score) else 0.0
    report.mean_score_normal = float(neg_score.mean()) if len(neg_score) else 0.0

    # The live operating point: exactly what the rule engine received.
    tp, fp, tn, fn = _confusion(samples, lambda s: s.flagged)
    report.true_positive, report.false_positive = tp, fp
    report.true_negative, report.false_negative = tn, fn
    report.precision, report.recall, report.f1 = _scores(tp, fp, tn, fn)
    report.specificity = tn / (tn + fp) if tn + fp else 0.0
    report.false_positive_rate = fp / (fp + tn) if fp + tn else 0.0

    # Sweep the same operating parameter the calibrator controls.
    best = (0.0, 1.0)
    for candidate in np.linspace(0.25, 6.0, 231):
        tp_c, fp_c, tn_c, fn_c = _confusion(samples, lambda s, c=float(candidate): ratio_of(s) > c)
        _p, _r, f1 = _scores(tp_c, fp_c, tn_c, fn_c)
        if f1 > best[0] + 1e-9:
            best = (f1, float(candidate))
    report.best_f1, report.best_ratio = best

    for candidate in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0):
        tp_c, fp_c, tn_c, fn_c = _confusion(
            samples, lambda s, c=candidate: ratio_of(s) > c
        )
        precision_c, recall_c, f1_c = _scores(tp_c, fp_c, tn_c, fn_c)
        report.operating_points.append(
            {
                "ratio": candidate,
                "precision": precision_c,
                "recall": recall_c,
                "f1": f1_c,
                "false_positive_rate": fp_c / (fp_c + tn_c) if fp_c + tn_c else 0.0,
                "flagged": tp_c + fp_c,
            }
        )

    by_camera: dict[str, list[Sample]] = {}
    by_kind: dict[str, list[Sample]] = {}
    for sample in samples:
        by_camera.setdefault(sample.camera_id, []).append(sample)
        kinds = sample.ground_truth or ("normal",)
        for kind in kinds:
            by_kind.setdefault(kind, []).append(sample)
    for camera_id, subset in by_camera.items():
        pos = np.array([ratio_of(s) for s in subset if s.ground_truth])
        neg = np.array([ratio_of(s) for s in subset if not s.ground_truth])
        report.per_camera[camera_id] = {
            "windows": len(subset),
            "anomalous": len(pos),
            "normal": len(neg),
            "auroc": _auroc(pos, neg),
            "mean_ratio": float(np.mean([ratio_of(s) for s in subset])),
            "threshold": float(np.median([s.threshold for s in subset])),
            "flagged": int(sum(1 for s in subset if s.flagged)),
        }
    for kind, subset in by_kind.items():
        report.per_kind[kind] = {
            "windows": len(subset),
            "mean_ratio": float(np.mean([ratio_of(s) for s in subset])),
            "share_flagged": float(np.mean([1.0 if s.flagged else 0.0 for s in subset])),
        }

    if engine is not None:
        snapshot = engine.snapshot()
        if snapshot is not None:
            report.latency_ms = snapshot.metrics.latency_ms
            report.fps = snapshot.metrics.fps
            report.duration_s = snapshot.metrics.uptime_s
            report.event_counts = dict(Counter(event.kind for event in snapshot.events))
    return report


def render_report(report: BenchmarkReport) -> str:
    lines = [
        "VISIONGUARD SYNTHETIC BENCHMARK",
        "=" * 66,
        f"windows scored      {report.windows:>8}",
        f"  anomalous         {report.anomalous:>8}",
        f"  normal            {report.normal:>8}",
        "",
        f"AUROC               {report.auroc:>8.3f}   (error / per-camera threshold)",
        f"mean ratio anomaly  {report.mean_ratio_anomalous:>8.2f}",
        f"mean ratio normal   {report.mean_ratio_normal:>8.2f}",
        f"p95 ratio normal    {report.p95_ratio_normal:>8.2f}",
        "",
        "operating point (hysteresis flag used by the rule engine)",
        f"  precision         {report.precision:>8.3f}",
        f"  recall            {report.recall:>8.3f}",
        f"  F1                {report.f1:>8.3f}",
        f"  specificity       {report.specificity:>8.3f}",
        f"  false positive    {report.false_positive_rate:>8.3f}",
        f"  confusion         tp={report.true_positive} fp={report.false_positive} "
        f"tn={report.true_negative} fn={report.false_negative}",
        f"best F1             {report.best_f1:>8.3f} @ ratio {report.best_ratio:.2f}",
        "",
        "operating point sweep (ratio = smoothed error / calibrated threshold)",
        "  ratio  precision  recall      F1     FPR  flagged",
    ]
    for row in report.operating_points:
        marker = " <- live" if abs(row["ratio"] - 1.0) < 1e-9 else ""
        lines.append(
            f"  {row['ratio']:>5.2f}   {row['precision']:>7.3f}  {row['recall']:>6.3f}  "
            f"{row['f1']:>6.3f}  {row['false_positive_rate']:>6.3f}  {int(row['flagged']):>7}{marker}"
        )
    lines += [
        "",
        f"throughput          {report.fps:>8.1f} fps (sum of camera streams)",
        f"latency             {report.latency_ms:>8.2f} ms/frame",
        "",
        "per kind (mean ratio / share of windows flagged)",
    ]
    for kind, values in sorted(report.per_kind.items()):
        lines.append(
            f"  {kind:<16}{values['mean_ratio']:>7.2f}  {values['share_flagged'] * 100:>5.0f}%  n={values['windows']}"
        )
    lines += ["", "per camera (AUROC / mean ratio / threshold / flagged)"]
    for camera, values in sorted(report.per_camera.items()):
        auroc = values["auroc"]
        auroc_text = f"{auroc:>5.2f}" if math.isfinite(auroc) else "  n/a"
        lines.append(
            f"  {camera:<8} AUROC {auroc_text}  ratio {values['mean_ratio']:>5.2f}  "
            f"thr {values['threshold']:>5.3f}  "
            f"flagged {values['flagged']:>3}/{values['windows']:<3}"
        )
    if report.event_counts:
        lines += ["", "events raised by the rule engine"]
        for kind, count in sorted(report.event_counts.items()):
            lines.append(f"  {kind:<20}{count:>4}")
    lines += ["", report.caveat]
    return "\n".join(lines)
