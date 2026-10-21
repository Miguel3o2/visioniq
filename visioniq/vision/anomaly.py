"""Sequence anomaly detection.

Two interchangeable autoencoder backends reconstruct a window of scene
features; anomalies are windows the model cannot reconstruct.

* ``torch`` - a genuine LSTM autoencoder (encoder LSTM -> latent -> decoder
  LSTM), matching the architecture described by the original project.
* ``numpy`` - a small MLP autoencoder implemented with hand-written
  backpropagation, so the system still works without PyTorch installed.

Both are trained **only on normal operation** during a warm-up period, then
frozen.  The decision threshold is calibrated from the reconstruction-error
distribution of that warm-up data (percentile + margin) instead of being a
magic constant.  A hysteresis counter stops isolated spikes from raising
alarms.
"""

from __future__ import annotations

import json
import logging
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from visioniq.config import ModelConfig
from visioniq.core.types import FEATURE_LABELS, AnomalyResult

log = logging.getLogger(__name__)

#: Bumped whenever the threshold calibration changes, so a cached model does
#: not silently keep an operating point produced by an older scheme.
CALIBRATION_VERSION = "ema-v2"


# ---------------------------------------------------------------------------
# NumPy MLP autoencoder
# ---------------------------------------------------------------------------

class _NumpyAE:
    def __init__(self, n_in: int, hidden: int, latent: int, seed: int = 0) -> None:
        self.n_in = n_in
        self.hidden = hidden
        self.latent = latent
        rng = np.random.default_rng(seed)
        dims = {"W1": (n_in, hidden), "W2": (hidden, latent), "W3": (latent, hidden), "W4": (hidden, n_in)}
        self.params: dict[str, np.ndarray] = {}
        for name, (fan_in, fan_out) in dims.items():
            limit = 1.0 / math.sqrt(fan_in)
            self.params[name] = rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)
            self.params[f"b{name[1]}"] = np.zeros((fan_out,), dtype=np.float32)
        self._velocity = {k: np.zeros_like(v) for k, v in self.params.items()}

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        p = self.params
        h1 = np.tanh(x @ p["W1"] + p["b1"])
        z = np.tanh(h1 @ p["W2"] + p["b2"])
        h3 = np.tanh(z @ p["W3"] + p["b3"])
        y = h3 @ p["W4"] + p["b4"]
        return y, (x, h1, z, h3)

    def fit(self, x: np.ndarray, *, epochs: int = 45, lr: float = 3e-3,
            batch_size: int = 32, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        p = self.params
        for _epoch in range(max(1, epochs)):
            order = rng.permutation(len(x))
            for start in range(0, len(x), batch_size):
                batch = x[order[start:start + batch_size]]
                y, (xb, h1, z, h3) = self._forward(batch)
                n = max(1, len(batch))
                d4 = np.clip(2.0 * (y - batch) / n, -8.0, 8.0)
                grads = {
                    "W4": h3.T @ d4,
                    "b4": d4.sum(axis=0),
                }
                d3 = (d4 @ p["W4"].T) * (1.0 - h3 * h3)
                grads["W3"] = z.T @ d3
                grads["b3"] = d3.sum(axis=0)
                d2 = (d3 @ p["W3"].T) * (1.0 - z * z)
                grads["W2"] = h1.T @ d2
                grads["b2"] = d2.sum(axis=0)
                d1 = (d2 @ p["W2"].T) * (1.0 - h1 * h1)
                grads["W1"] = xb.T @ d1
                grads["b1"] = d1.sum(axis=0)
                for key, grad in grads.items():
                    grad = np.clip(grad, -5.0, 5.0)
                    self._velocity[key] = 0.9 * self._velocity[key] - lr * grad
                    p[key] += self._velocity[key]

    def error(self, x: np.ndarray) -> np.ndarray:
        y, _ = self._forward(x)
        return np.mean((y - x) ** 2, axis=1)

    def feature_error(self, x: np.ndarray) -> np.ndarray:
        """Per-input-value squared reconstruction error (n, n_in)."""

        y, _ = self._forward(x)
        return (y - x) ** 2

    def state(self) -> dict[str, np.ndarray]:
        return dict(self.params)

    def load_state(self, state: dict[str, np.ndarray]) -> None:
        for key in self.params:
            if key in state:
                self.params[key] = np.asarray(state[key], dtype=np.float32)
        self._velocity = {k: np.zeros_like(v) for k, v in self.params.items()}


# ---------------------------------------------------------------------------
# PyTorch LSTM autoencoder
# ---------------------------------------------------------------------------

class _TorchAE:
    def __init__(self, n_in: int, hidden: int, latent: int, seed: int = 0) -> None:
        import torch
        from torch import nn

        torch.set_num_threads(1)
        torch.manual_seed(seed)
        self._torch = torch
        self._nn = nn
        self.n_in = n_in
        self.hidden = hidden

        class LSTMAE(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = nn.LSTM(n_in, hidden, batch_first=True)
                self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
                self.head = nn.Linear(hidden, n_in)

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # noqa: F821
                _, (state, _) = self.encoder(x)
                latent = state[-1].unsqueeze(1).expand(-1, x.size(1), -1)
                decoded, _ = self.decoder(latent)
                return self.head(decoded)

        self.net = LSTMAE()

    def fit(self, x: np.ndarray, *, epochs: int = 14, lr: float = 2e-3,
            batch_size: int = 32, seed: int = 0) -> None:
        torch = self._torch
        torch.manual_seed(seed)
        data = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        optimiser = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-5)
        loss_fn = self._nn.MSELoss()
        self.net.train()
        for _epoch in range(max(1, epochs)):
            order = torch.randperm(len(data))
            for start in range(0, len(data), batch_size):
                batch = data[order[start:start + batch_size]]
                optimiser.zero_grad(set_to_none=True)
                loss = loss_fn(self.net(batch), batch)
                loss.backward()
                self._nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimiser.step()
        self.net.eval()

    def _errors(self, x: np.ndarray) -> np.ndarray:
        torch = self._torch
        data = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        with torch.no_grad():
            reconstructed = self.net(data)
            squared = (reconstructed - data) ** 2
        return squared.mean(dim=1).numpy()  # (n_windows, n_features)

    def error(self, x: np.ndarray) -> np.ndarray:
        per_feature = self._errors(x)
        return per_feature.mean(axis=1)

    def feature_error(self, x: np.ndarray) -> np.ndarray:
        return self._errors(x)

    def state(self) -> dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy() for k, v in self.net.state_dict().items()}

    def load_state(self, state: dict[str, np.ndarray]) -> None:
        torch = self._torch
        self.net.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in state.items()})
        self.net.eval()


# ---------------------------------------------------------------------------
# Public autoencoder wrapper
# ---------------------------------------------------------------------------

class SequenceAutoencoder:
    """Standardisation + backend + persistence."""

    def __init__(self, n_features: int, window: int, cfg: ModelConfig, *, seed: int = 0) -> None:
        self.n_features = n_features
        self.window = window
        self.cfg = cfg
        self.seed = seed
        self.mean = np.zeros(n_features, dtype=np.float32)
        self.std = np.ones(n_features, dtype=np.float32)
        self.var = np.ones(n_features, dtype=np.float32)
        self._backend_name = "numpy"
        self._backend: Any = None
        self._make_backend(cfg.backend)

    def _make_backend(self, backend: str) -> None:
        if backend in {"auto", "torch"}:
            try:
                self._backend = _TorchAE(self.n_features, self.cfg.hidden, max(4, self.cfg.hidden // 2), self.seed)
                self._backend_name = "torch-lstm-ae"
                return
            except Exception as exc:  # pragma: no cover - depends on install
                if backend == "torch":
                    raise
                log.info("PyTorch unavailable (%s); using NumPy autoencoder", exc)
        self._backend = _NumpyAE(
            self.n_features * self.window,
            max(24, self.cfg.hidden * 2),
            max(8, self.cfg.hidden // 2),
            self.seed,
        )
        self._backend_name = "numpy-mlp-ae"

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def _standardise(self, windows: np.ndarray) -> np.ndarray:
        scaled = (windows - self.mean) / self.std
        if self.cfg.input_clip > 0:
            np.clip(scaled, -self.cfg.input_clip, self.cfg.input_clip, out=scaled)
        return scaled.astype(np.float32)

    def adapt_scaler(self, feature_row: np.ndarray, alpha: float = 0.0015) -> None:
        """Slowly track scene-level drift (more traffic at midday, lighting).

        Anomalies are brief by nature, so a very slow exponential update is
        dominated by normal operation.  Without this, a busy afternoon looks
        anomalous to a model calibrated on a quiet morning - the most common
        failure mode of reconstruction-based anomaly detection.
        """
        row = np.nan_to_num(feature_row.astype(np.float32), nan=0.0)
        delta = row - self.mean
        self.mean += alpha * delta
        self.var = np.maximum(self.var + alpha * (delta * delta - self.var), 0.0144)
        self.std = np.sqrt(self.var).astype(np.float32)

    def _flatten(self, windows: np.ndarray) -> np.ndarray:
        if self._backend_name == "numpy-mlp-ae":
            return windows.reshape(len(windows), -1)
        return windows

    def fit(self, windows: np.ndarray) -> None:
        data = np.asarray(windows, dtype=np.float32)
        if data.ndim != 3 or data.shape[1:] != (self.window, self.n_features):
            raise ValueError(f"expected (n, {self.window}, {self.n_features}) windows, got {data.shape}")
        self.mean = data.mean(axis=(0, 1)).astype(np.float32)
        # A floor on the scale is important: features such as "vehicles" are
        # constant during calibration, and dividing by ~0 would turn a single
        # later vehicle into a huge z-score.  The features are already scaled
        # to comparable ranges, so an absolute floor is appropriate.
        self.std = np.maximum(data.std(axis=(0, 1)), 0.12).astype(np.float32)
        self.var = np.square(self.std)
        standardised = self._standardise(data)
        params = dict(
            epochs=self.cfg.epochs,
            lr=self.cfg.learning_rate,
            batch_size=self.cfg.batch_size,
            seed=self.seed,
        )
        self._backend.fit(self._flatten(standardised), **params)

    def error(self, windows: np.ndarray) -> np.ndarray:
        data = np.asarray(windows, dtype=np.float32)
        return self._backend.error(self._flatten(self._standardise(data)))

    def feature_error(self, windows: np.ndarray) -> np.ndarray:
        data = np.asarray(windows, dtype=np.float32)
        per_value = self._backend.feature_error(self._flatten(self._standardise(data)))
        if self._backend_name == "numpy-mlp-ae":
            # The MLP reconstructs a flattened window; fold the time axis back
            # so attribution lines up with FEATURE_NAMES.
            per_value = per_value.reshape(len(data), self.window, self.n_features).mean(axis=1)
        return per_value

    # -- persistence -------------------------------------------------------
    def save(self, path: Path, *, threshold: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "n_features": self.n_features,
            "window": self.window,
            "hidden": self.cfg.hidden,
            "backend": self._backend_name,
            "input_clip": self.cfg.input_clip,
            "calibration": CALIBRATION_VERSION,
            "threshold": threshold,
        }
        np.savez_compressed(
            path,
            meta=np.array(json.dumps(meta)),
            mean=self.mean,
            std=self.std,
            **{f"param_{k}": v for k, v in self._backend.state().items()},
        )

    @classmethod
    def load(cls, path: Path, cfg: ModelConfig) -> tuple["SequenceAutoencoder", float] | None:
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as archive:
                meta = json.loads(str(archive["meta"]))
                model = cls(int(meta["n_features"]), int(meta["window"]), cfg, seed=0)
                if meta.get("backend") != model.backend_name:
                    log.info("model backend changed (%s -> %s); recalibrating",
                             meta.get("backend"), model.backend_name)
                    return None
                if float(meta.get("input_clip", 0.0)) != float(cfg.input_clip):
                    log.info("model input clipping changed (%s -> %s); recalibrating",
                             meta.get("input_clip"), cfg.input_clip)
                    return None
                if meta.get("calibration") != CALIBRATION_VERSION:
                    log.info("model calibration scheme changed; recalibrating")
                    return None
                if int(meta["window"]) != cfg.window or int(meta["n_features"]) != model.n_features:
                    return None
                state = {
                    key[len("param_"):]: archive[key]
                    for key in archive.files
                    if key.startswith("param_")
                }
                model.mean = np.asarray(archive["mean"], dtype=np.float32)
                model.std = np.asarray(archive["std"], dtype=np.float32)
                model.var = np.square(model.std)
                model._backend.load_state(state)
                return model, float(meta["threshold"])
        except Exception as exc:  # pragma: no cover - corrupt cache file
            log.warning("could not load model %s: %s", path, exc)
            return None


# ---------------------------------------------------------------------------
# Online scorer
# ---------------------------------------------------------------------------

class AnomalyScorer:
    """Turns per-frame features into calibrated anomaly scores."""

    #: Smoothing applied to the live reconstruction error.  Calibration must
    #: use the *same* smoothing: the 97th percentile of raw errors is much
    #: higher than the 97th percentile of smoothed errors, and using the
    #: former as a threshold silently desensitises noisy cameras.
    EMA_ALPHA = 0.4

    def __init__(self, cfg: ModelConfig, camera_id: str, *, seed: int = 0,
                 model_path: Path | None = None) -> None:
        self.cfg = cfg
        self.camera_id = camera_id
        self.model_path = model_path
        self._window = cfg.window
        self._sampled: deque[np.ndarray] = deque(maxlen=cfg.window)
        self._frame = 0
        self._warmup: list[np.ndarray] = []
        self._model: SequenceAutoencoder | None = None
        self._threshold = 0.0
        self._ema: float | None = None
        self._above = 0
        self._below = 0
        self._flag = False
        self._windows = 0
        self._last = AnomalyResult.warmup(0.0)
        self._normal_history: deque[np.ndarray] = deque(maxlen=240)
        self._load(model_path)

    # -- persistence -------------------------------------------------------
    def _load(self, path: Path | None) -> None:
        if path is None:
            return
        loaded = SequenceAutoencoder.load(path, self.cfg)
        if loaded is None:
            return
        model, threshold = loaded
        self._model = model
        self._threshold = max(
            threshold,
            self.cfg.error_floor * self.cfg.threshold_multiplier,
            1e-6,
        )
        log.info("%s: loaded model from %s (%s, threshold %.4f)",
                 self.camera_id, path.name, model.backend_name, threshold)

    def _save(self) -> None:
        if self.model_path is None or self._model is None:
            return
        try:
            self._model.save(self.model_path, threshold=self._threshold)
        except OSError as exc:
            log.warning("%s: could not save model: %s", self.camera_id, exc)

    @property
    def backend_name(self) -> str:
        return self._model.backend_name if self._model else "pending"

    @property
    def calibrated(self) -> bool:
        return self._model is not None

    @property
    def threshold(self) -> float:
        return self._threshold

    # -- inference ---------------------------------------------------------
    def observe(self, features: np.ndarray) -> AnomalyResult:
        features = np.nan_to_num(features.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        self._frame += 1
        if self._frame % self.cfg.stride != 0:
            return self._last
        self._sampled.append(features)

        if len(self._sampled) < self._window:
            progress = 0.05 + 0.2 * (len(self._sampled) / self._window)
            self._last = AnomalyResult.warmup(progress)
            return self._last

        window = np.stack(self._sampled)

        if self._model is None:
            self._warmup.append(window)
            needed = max(1, self.cfg.warmup_windows)
            if len(self._warmup) < needed:
                progress = 0.25 + 0.7 * (len(self._warmup) / needed)
                self._last = AnomalyResult.warmup(progress)
                return self._last
            self._calibrate()
            # fall through and score with the freshly calibrated model

        return self._score(window)

    def _calibrate(self) -> None:
        data = np.stack(self._warmup)
        # Calibrate the threshold on windows the model has NOT trained on.
        # Training-set reconstruction error is always optimistic; using it
        # would produce a threshold that normal traffic exceeds immediately.
        split = max(8, int(len(data) * 0.75))
        train, validation = data[:split], data[split:]
        if len(validation) < 4:
            train, validation = data, data
        model = SequenceAutoencoder(data.shape[2], self._window, self.cfg, seed=0)
        model.fit(train)
        errors = _ema(errors=model.error(validation), alpha=self.EMA_ALPHA)
        percentile = float(np.percentile(errors, self.cfg.threshold_percentile))
        median = float(np.median(errors))
        mad = float(np.median(np.abs(errors - median))) * 1.4826
        self._threshold = max(
            percentile,
            median + 3.0 * mad,
            self.cfg.error_floor,
        ) * self.cfg.threshold_multiplier
        self._model = model
        self._normal_history.extend(self._warmup)
        self._warmup.clear()
        log.info(
            "%s: calibrated %s on %d windows (%d held out), threshold=%.4f (median=%.4f)",
            self.camera_id, model.backend_name, len(train), len(validation),
            self._threshold, median,
        )
        self._save()

    def _score(self, window: np.ndarray) -> AnomalyResult:
        assert self._model is not None
        batch = window[None, :, :]
        raw = float(self._model.error(batch)[0])
        alpha = self.EMA_ALPHA
        self._ema = raw if self._ema is None else alpha * raw + (1 - alpha) * self._ema
        ratio = self._ema / max(self._threshold, 1e-9)
        score = 1.0 / (1.0 + math.exp(-4.5 * (ratio - 1.0)))
        self._windows += 1

        if ratio > 1.0:
            self._above += 1
            self._below = 0
        else:
            self._below += 1
            if self._below >= 2:
                self._above = 0
        if self._above >= self.cfg.hysteresis:
            self._flag = True
        elif self._below >= 2:
            self._flag = False

        if not self._flag:
            self._model.adapt_scaler(window[-1])

        attribution: tuple[tuple[str, float], ...] = ()
        if score > 0.5:
            per_feature = self._model.feature_error(batch)[0]
            total = float(per_feature.sum()) + 1e-9
            ranked = np.argsort(per_feature)[::-1][:3]
            attribution = tuple(
                (FEATURE_LABELS.get(_feature_name(i), f"f{i}"), float(per_feature[i] / total))
                for i in ranked
            )

        if self.cfg.adaptation and not self._flag and self._windows % 240 == 0:
            self._adapt(window)

        self._last = AnomalyResult(
            score=float(score),
            raw_error=self._ema,
            threshold=self._threshold,
            is_anomaly=self._flag,
            calibrating=False,
            progress=1.0,
            top_features=attribution,
            windows=self._windows,
        )
        return self._last

    def _adapt(self, window: np.ndarray) -> None:
        self._normal_history.append(window)
        if len(self._normal_history) < 16:
            return
        data = np.stack(list(self._normal_history))
        assert self._model is not None
        try:
            self._model.fit(data)
            errors = _ema(errors=self._model.error(data), alpha=self.EMA_ALPHA)
            median = float(np.median(errors))
            mad = float(np.median(np.abs(errors - median))) * 1.4826
            self._threshold = max(
                median + 3.0 * mad,
                self.cfg.error_floor,
            ) * self.cfg.threshold_multiplier
            self._ema = None
            self._save()
        except Exception as exc:  # pragma: no cover - adaptation is best effort
            log.warning("%s: adaptation failed: %s", self.camera_id, exc)

    def model_info(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "threshold": self._threshold,
            "windows": self._windows,
            "calibrated": self.calibrated,
        }


def _ema(errors: np.ndarray, alpha: float) -> np.ndarray:
    """Apply the live error smoother to a sequence of calibration errors."""

    values = np.asarray(errors, dtype=np.float64)
    if values.size == 0:
        return values
    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    for index in range(1, len(values)):
        smoothed[index] = alpha * values[index] + (1.0 - alpha) * smoothed[index - 1]
    return smoothed


def _feature_name(index: int) -> str:
    from visioniq.core.types import FEATURE_NAMES
    return FEATURE_NAMES[index] if 0 <= index < len(FEATURE_NAMES) else f"f{index}"
