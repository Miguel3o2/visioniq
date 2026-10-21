from __future__ import annotations

import numpy as np

from visioniq.config import ModelConfig
from visioniq.vision.anomaly import AnomalyScorer, SequenceAutoencoder, _ema


def _windows(n: int, window: int, features: int = 12, scale: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, scale, (n, window, features)).astype(np.float32)
    return base


def _cfg(**overrides) -> ModelConfig:
    values = dict(
        backend="numpy",
        window=8,
        stride=2,
        hidden=16,
        warmup_windows=40,
        epochs=3,
        batch_size=8,
        threshold_percentile=97.0,
        input_clip=4.0,
        hysteresis=2,
    )
    values.update(overrides)
    return ModelConfig(**values)


def test_autoencoder_learns_normal_and_scores_outlier_higher():
    cfg = _cfg()
    normal = _windows(80, cfg.window)
    model = SequenceAutoencoder(12, cfg.window, cfg, seed=1)
    model.fit(normal)
    quiet = model.error(normal[:10])
    loud = normal[:10].copy()
    loud[:, :, 0] += 6.0
    assert float(model.error(loud).mean()) > float(quiet.mean()) * 3.0

    per_feature = model.feature_error(loud)
    assert per_feature.shape == (10, 12)
    assert np.argmax(per_feature.mean(axis=0)) == 0


def _feed(scorer: AnomalyScorer, rows: np.ndarray) -> None:
    for row in rows:
        scorer.observe(row)


def test_threshold_floor_prevents_static_scene_blowup(tmp_path):
    cfg = _cfg(warmup_windows=24, error_floor=0.2, threshold_multiplier=1.0)
    scorer = AnomalyScorer(cfg, "CAM-X", model_path=tmp_path / "x.npz")
    window = np.zeros((cfg.window, 12), dtype=np.float32)
    needed = (cfg.warmup_windows + cfg.window + 2) * cfg.stride
    _feed(scorer, np.repeat(window[:1], needed, axis=0))
    assert scorer.calibrated
    assert scorer.threshold >= 0.2


def test_scorer_flags_sustained_anomaly(tmp_path):
    cfg = _cfg(hysteresis=3)
    scorer = AnomalyScorer(cfg, "CAM-Y", model_path=tmp_path / "y.npz")
    normal_rows = _windows(cfg.warmup_windows + 20, cfg.window, seed=5).reshape(-1, 12)
    _feed(scorer, normal_rows)
    assert scorer.calibrated

    spike_rows = normal_rows[-cfg.window:].copy()
    spike_rows[:, 1] += 8.0
    for _ in range(4):
        _feed(scorer, spike_rows)
    result = scorer.observe(spike_rows[0])
    assert result.score > 0.5
    assert result.is_anomaly



def test_calibration_smoothing_matches_live_ema():
    errors = np.array([0.0, 10.0, 0.0, 0.0], dtype=np.float64)
    smoothed = _ema(errors, alpha=0.5)
    assert smoothed[0] == 0.0
    assert np.allclose(smoothed, [0.0, 5.0, 2.5, 1.25])
    assert smoothed.max() < errors.max()
    assert _ema(np.array([], dtype=float), alpha=0.5).size == 0


def test_save_and_load_roundtrip(tmp_path):
    cfg = _cfg()
    model = SequenceAutoencoder(12, cfg.window, cfg, seed=2)
    model.fit(_windows(50, cfg.window, seed=3))
    path = tmp_path / "model.npz"
    model.save(path, threshold=0.42)

    loaded = SequenceAutoencoder.load(path, cfg)
    assert loaded is not None
    restored, threshold = loaded
    assert threshold == 0.42
    sample = _windows(4, cfg.window, seed=4)
    assert np.allclose(model.error(sample), restored.error(sample), atol=1e-5)


def test_scaler_adaptation_tracks_drift():
    cfg = _cfg()
    model = SequenceAutoencoder(12, cfg.window, cfg)
    model.fit(_windows(40, cfg.window, scale=0.02))
    before = model.mean.copy()
    row = np.full(12, 2.0, dtype=np.float32)
    for _ in range(400):
        model.adapt_scaler(row, alpha=0.01)
    assert float(np.linalg.norm(model.mean - before)) > 1.0
    assert float(model.std.min()) >= 0.12
