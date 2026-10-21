"""Configuration for VisionIQ.

A TOML file is optional; sane defaults matching the original eight-camera
deployment ship with the package.  Unknown keys are rejected so typos in a
deployment file fail loudly instead of silently doing nothing.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

DEFAULT_AFTER_HOURS = (20.0, 7.0)


class ConfigError(ValueError):
    """Raised when a configuration file is malformed."""


@dataclass(slots=True, frozen=True)
class Rect:
    """Normalised rectangle in frame coordinates (0..1)."""

    x: float
    y: float
    w: float
    h: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.x <= 1.0 and 0.0 <= self.y <= 1.0):
            raise ConfigError(f"rect origin outside frame: {self}")
        if not (0.0 < self.w <= 1.0 and 0.0 < self.h <= 1.0):
            raise ConfigError(f"rect size must be positive: {self}")
        if self.x + self.w > 1.0 + 1e-6 or self.y + self.h > 1.0 + 1e-6:
            raise ConfigError(f"rect exceeds frame bounds: {self}")

    @classmethod
    def parse(cls, raw: Any) -> "Rect":
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            raise ConfigError(f"expected [x, y, w, h], got {raw!r}")
        try:
            x, y, w, h = (float(v) for v in raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"invalid rect values: {raw!r}") from exc
        return cls(x, y, w, h)

    def contains(self, cx: float, cy: float) -> bool:
        return self.x <= cx <= self.x + self.w and self.y <= cy <= self.y + self.h


@dataclass(slots=True, frozen=True)
class ZoneConfig:
    name: str
    rect: Rect
    kind: str = "general"  # general | entry | restricted


@dataclass(slots=True)
class CameraConfig:
    id: str
    name: str
    zone: str
    source: str = "synthetic:lobby"
    tint: tuple[int, int, int] = (0, 180, 255)
    restricted: bool = False
    access_controlled: bool = False
    capacity: int = 6
    loiter_seconds: float = 180.0
    after_hours: tuple[float, float] = DEFAULT_AFTER_HOURS
    seed: int = 0
    fps: float = 10.0
    zones: tuple[ZoneConfig, ...] = ()
    # Synthetic scenes only: how busy the scene is during clean operation.
    traffic: float = 1.0

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ConfigError("camera id must not be empty")
        if self.fps <= 0 or self.fps > 60:
            raise ConfigError(f"{self.id}: fps must be in (0, 60]")
        if self.capacity < 1:
            raise ConfigError(f"{self.id}: capacity must be >= 1")
        if self.loiter_seconds <= 0:
            raise ConfigError(f"{self.id}: loiter_seconds must be > 0")
        start, end = self.after_hours
        if not (0 <= start <= 24 and 0 <= end <= 24):
            raise ConfigError(f"{self.id}: after_hours must be hours within 0..24")
        if len(self.tint) != 3 or any(not 0 <= c <= 255 for c in self.tint):
            raise ConfigError(f"{self.id}: tint must be three 0..255 channels")

    @property
    def entry_zone(self) -> Rect | None:
        for zone in self.zones:
            if zone.kind == "entry":
                return zone.rect
        return None

    @property
    def restricted_zones(self) -> tuple[ZoneConfig, ...]:
        return tuple(z for z in self.zones if z.kind == "restricted")

    def is_after_hours(self, hour: float) -> bool:
        start, end = self.after_hours
        if start == end:
            return False
        if start < end:
            return start <= hour < end
        return hour >= start or hour < end


@dataclass(slots=True)
class ModelConfig:
    backend: str = "auto"  # auto | torch | numpy
    window: int = 24
    stride: int = 2
    hidden: int = 32
    warmup_windows: int = 400
    epochs: int = 12
    learning_rate: float = 2e-3
    batch_size: int = 32
    threshold_percentile: float = 97.0
    threshold_multiplier: float = 1.10
    # Standardised reconstruction errors are mean squared errors over a
    # window x feature matrix.  A threshold below this floor is not credible:
    # it means the calibration scene was almost perfectly static and the
    # 97th percentile collapsed towards zero.  The floor keeps a quiet camera
    # from declaring every later flicker an anomaly.
    error_floor: float = 0.05
    # Standardised features are clipped before entering the autoencoder.
    # Extremely rare-but-normal values (a crowd dispersing once) otherwise
    # dominate the mean squared error and inflate the threshold for the whole
    # camera.  Clipping keeps the model focused on typical deviations.
    input_clip: float = 4.0
    hysteresis: int = 2
    cooldown: float = 45.0
    adaptation: bool = False

    def __post_init__(self) -> None:
        if self.backend not in {"auto", "torch", "numpy"}:
            raise ConfigError("model.backend must be auto, torch or numpy")
        if self.window < 4:
            raise ConfigError("model.window must be >= 4")
        if self.stride < 1:
            raise ConfigError("model.stride must be >= 1")
        if self.hidden < 4:
            raise ConfigError("model.hidden must be >= 4")
        if not 1 <= self.warmup_windows <= 2000:
            raise ConfigError("model.warmup_windows must be in 1..2000")
        if not 50.0 <= self.threshold_percentile < 100.0:
            raise ConfigError("model.threshold_percentile must be in [50, 100)")
        if self.error_floor < 0.0:
            raise ConfigError("model.error_floor must be >= 0")
        if not 1.0 <= self.input_clip <= 20.0:
            raise ConfigError("model.input_clip must be in 1..20")
        if self.hysteresis < 1:
            raise ConfigError("model.hysteresis must be >= 1")


@dataclass(slots=True)
class RuleConfig:
    badge_window: float = 12.0
    tailgate_window: float = 6.0
    abandoned_seconds: float = 25.0
    anomaly_cooldown: float = 45.0
    event_cooldown: float = 60.0
    alert_retention_minutes: float = 30.0
    enabled: tuple[str, ...] = (
        "anomaly",
        "crowding",
        "loitering",
        "unbadged_access",
        "tailgate",
        "after_hours",
        "abandoned",
        "vehicle_restricted",
        "source_stall",
    )

    def has(self, name: str) -> bool:
        return name in self.enabled


@dataclass(slots=True)
class AppConfig:
    cameras: list[CameraConfig] = field(default_factory=list)
    model: ModelConfig = field(default_factory=ModelConfig)
    rules: RuleConfig = field(default_factory=RuleConfig)
    db_path: str = "data/visioniq.db"
    model_dir: str = "data/models"
    snapshot_dir: str = "snapshots"
    render_fps: float = 8.0
    display_width: int = 240
    display_height: int = 135
    retention_events: int = 500

    def validate(self) -> "AppConfig":
        if not self.cameras:
            raise ConfigError("at least one camera must be configured")
        seen: set[str] = set()
        for cam in self.cameras:
            if cam.id in seen:
                raise ConfigError(f"duplicate camera id: {cam.id}")
            seen.add(cam.id)
        return self


# ---------------------------------------------------------------------------
# Default deployment: the eight cameras from the original mock-up.
# ---------------------------------------------------------------------------

def _zones(entry: tuple[float, float, float, float] | None, restricted: tuple[float, float, float, float] | None = None) -> tuple[ZoneConfig, ...]:
    zones: list[ZoneConfig] = []
    if entry:
        zones.append(ZoneConfig("entrance", Rect.parse(entry), "entry"))
    if restricted:
        zones.append(ZoneConfig("secure area", Rect.parse(restricted), "restricted"))
    return tuple(zones)


DEFAULT_CAMERAS: tuple[CameraConfig, ...] = (
    CameraConfig(
        id="CAM-01", name="Main Entrance", zone="Ground / Zone A",
        source="synthetic:lobby", tint=(0, 210, 255),
        restricted=True, access_controlled=True, capacity=8,
        seed=11, traffic=1.2,
        zones=_zones((0.74, 0.30, 0.24, 0.55), (0.05, 0.55, 0.45, 0.35)),
    ),
    CameraConfig(
        id="CAM-02", name="Car Park B", zone="Exterior / Zone B",
        source="synthetic:parking", tint=(70, 200, 255),
        capacity=4, seed=22, traffic=1.0,
        zones=_zones((0.00, 0.55, 0.16, 0.40)),
    ),
    CameraConfig(
        id="CAM-03", name="Server Room", zone="Level 2 / Zone C",
        source="synthetic:server", tint=(0, 255, 190),
        restricted=True, capacity=2, loiter_seconds=90.0,
        seed=33, traffic=0.35,
        zones=_zones((0.42, 0.30, 0.16, 0.50), (0.10, 0.55, 0.80, 0.35)),
    ),
    CameraConfig(
        id="CAM-04", name="Executive Floor", zone="Level 3 / Zone D",
        source="synthetic:office", tint=(120, 180, 255),
        restricted=True, access_controlled=True, capacity=3,
        loiter_seconds=120.0, seed=44, traffic=0.7,
        zones=_zones((0.78, 0.28, 0.18, 0.55), (0.15, 0.55, 0.55, 0.33)),
    ),
    CameraConfig(
        id="CAM-05", name="Loading Bay", zone="Ground / Zone E",
        source="synthetic:loading", tint=(255, 190, 60),
        capacity=6, seed=55, traffic=0.9,
        zones=_zones((0.00, 0.50, 0.18, 0.45)),
    ),
    CameraConfig(
        id="CAM-06", name="Lobby West", zone="Ground / Zone F",
        source="synthetic:lobby", tint=(0, 230, 255),
        capacity=10, seed=66, traffic=1.5,
        zones=_zones((0.70, 0.30, 0.28, 0.55)),
    ),
    CameraConfig(
        id="CAM-07", name="Stairwell C", zone="Level 1 / Zone G",
        source="synthetic:stairwell", tint=(160, 210, 255),
        restricted=True, capacity=3, loiter_seconds=120.0,
        seed=77, traffic=0.45,
        zones=_zones((0.30, 0.35, 0.18, 0.50)),
    ),
    CameraConfig(
        id="CAM-08", name="CCTV Control Room", zone="Level 1 / Zone H",
        source="synthetic:office", tint=(0, 255, 220),
        restricted=True, access_controlled=True, capacity=2,
        loiter_seconds=240.0, seed=88, traffic=0.4,
        zones=_zones((0.20, 0.30, 0.16, 0.50)),
    ),
)


def default_config() -> AppConfig:
    return AppConfig(cameras=[replace(c) for c in DEFAULT_CAMERAS]).validate()


# ---------------------------------------------------------------------------
# TOML loading
# ---------------------------------------------------------------------------

_TOP_KEYS = {"app", "model", "rules", "cameras"}
_MISSING = object()


def _get(table: dict[str, Any], key: str, default: Any, where: str) -> Any:
    value = table.get(key, _MISSING)
    return default if value is _MISSING else value


def _check_keys(table: dict[str, Any], allowed: Iterable[str], where: str) -> None:
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        raise ConfigError(f"{where}: unknown key(s): {', '.join(unknown)}")


def _coerce(value: Any, kind: type, where: str) -> Any:
    try:
        if kind is bool:
            if not isinstance(value, bool):
                raise TypeError
            return value
        if kind is int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError
            return value
        if kind is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError
            return float(value)
        if kind is str:
            if not isinstance(value, str):
                raise TypeError
            return value
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where}: expected {kind.__name__}, got {value!r}") from exc
    return value


def _load_camera(raw: dict[str, Any], index: int) -> CameraConfig:
    where = f"cameras[{index}]"
    _check_keys(
        raw,
        {
            "id", "name", "zone", "source", "tint", "restricted",
            "access_controlled", "capacity", "loiter_seconds", "after_hours",
            "seed", "fps", "zones", "traffic",
        },
        where,
    )
    if "id" not in raw or "name" not in raw:
        raise ConfigError(f"{where}: id and name are required")
    cid = _coerce(raw["id"], str, f"{where}.id")
    where = f"camera {cid}"

    zones: list[ZoneConfig] = []
    for zi, zone in enumerate(raw.get("zones", [])):
        if not isinstance(zone, dict):
            raise ConfigError(f"{where}.zones[{zi}]: expected a table")
        _check_keys(zone, {"name", "rect", "kind"}, f"{where}.zones[{zi}]")
        kind = zone.get("kind", "general")
        if kind not in {"general", "entry", "restricted"}:
            raise ConfigError(f"{where}.zones[{zi}]: kind must be entry, restricted or general")
        zones.append(
            ZoneConfig(
                name=_coerce(zone.get("name", kind), str, f"{where}.zones[{zi}].name"),
                rect=Rect.parse(zone.get("rect")),
                kind=kind,
            )
        )

    tint = raw.get("tint", (0, 190, 255))
    if not isinstance(tint, (list, tuple)) or len(tint) != 3:
        raise ConfigError(f"{where}.tint: expected [r, g, b]")

    after_hours = raw.get("after_hours", DEFAULT_AFTER_HOURS)
    if not isinstance(after_hours, (list, tuple)) or len(after_hours) != 2:
        raise ConfigError(f"{where}.after_hours: expected [start_hour, end_hour]")

    camera = CameraConfig(
        id=cid,
        name=_coerce(raw["name"], str, f"{where}.name"),
        zone=_coerce(raw.get("zone", "Unzoned"), str, f"{where}.zone"),
        source=_coerce(raw.get("source", "synthetic:lobby"), str, f"{where}.source"),
        tint=tuple(int(_coerce(c, float, f"{where}.tint")) for c in tint),  # type: ignore[arg-type]
        restricted=_coerce(raw.get("restricted", False), bool, f"{where}.restricted"),
        access_controlled=_coerce(raw.get("access_controlled", False), bool, f"{where}.access_controlled"),
        capacity=_coerce(raw.get("capacity", 6), int, f"{where}.capacity"),
        loiter_seconds=_coerce(raw.get("loiter_seconds", 180.0), float, f"{where}.loiter_seconds"),
        after_hours=(float(after_hours[0]), float(after_hours[1])),
        seed=_coerce(raw.get("seed", index), int, f"{where}.seed"),
        fps=_coerce(raw.get("fps", 10.0), float, f"{where}.fps"),
        zones=tuple(zones),
        traffic=_coerce(raw.get("traffic", 1.0), float, f"{where}.traffic"),
    )
    return camera


def load_config(path: str | Path | None) -> AppConfig:
    """Load a TOML config, falling back to the built-in deployment."""

    if path is None:
        return default_config()
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    try:
        with p.open("rb") as fh:
            raw = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{p}: invalid TOML: {exc}") from exc

    _check_keys(raw, _TOP_KEYS, str(p))
    config = default_config()

    app_raw = raw.get("app", {})
    if not isinstance(app_raw, dict):
        raise ConfigError("app: expected a table")
    _check_keys(
        app_raw,
        {"db_path", "model_dir", "snapshot_dir", "render_fps", "display_width", "display_height", "retention_events"},
        "app",
    )
    for key in ("db_path", "model_dir", "snapshot_dir"):
        if key in app_raw:
            setattr(config, key, _coerce(app_raw[key], str, f"app.{key}"))
    for key in ("render_fps", "display_width", "display_height"):
        if key in app_raw:
            setattr(config, key, _coerce(app_raw[key], float if key == "render_fps" else int, f"app.{key}"))
    if "retention_events" in app_raw:
        config.retention_events = _coerce(app_raw["retention_events"], int, "app.retention_events")

    model_raw = raw.get("model", {})
    if not isinstance(model_raw, dict):
        raise ConfigError("model: expected a table")
    _check_keys(
        model_raw,
        {
            "backend", "window", "stride", "hidden", "warmup_windows", "epochs",
            "learning_rate", "batch_size", "threshold_percentile", "threshold_multiplier",
            "error_floor", "input_clip", "hysteresis", "cooldown", "adaptation",
        },
        "model",
    )
    model = config.model
    for key in ("backend", "window", "stride", "hidden", "warmup_windows", "epochs", "batch_size", "hysteresis"):
        if key in model_raw:
            kind = str if key == "backend" else int
            setattr(model, key, _coerce(model_raw[key], kind, f"model.{key}"))
    for key in ("learning_rate", "threshold_percentile", "threshold_multiplier", "error_floor", "input_clip", "cooldown"):
        if key in model_raw:
            setattr(model, key, _coerce(model_raw[key], float, f"model.{key}"))
    if "adaptation" in model_raw:
        model.adaptation = _coerce(model_raw["adaptation"], bool, "model.adaptation")

    rules_raw = raw.get("rules", {})
    if not isinstance(rules_raw, dict):
        raise ConfigError("rules: expected a table")
    _check_keys(
        rules_raw,
        {"badge_window", "tailgate_window", "abandoned_seconds", "anomaly_cooldown", "event_cooldown", "alert_retention_minutes", "enabled"},
        "rules",
    )
    rules = config.rules
    for key in ("badge_window", "tailgate_window", "abandoned_seconds", "anomaly_cooldown", "event_cooldown", "alert_retention_minutes"):
        if key in rules_raw:
            setattr(rules, key, _coerce(rules_raw[key], float, f"rules.{key}"))
    if "enabled" in rules_raw:
        enabled = rules_raw["enabled"]
        if not isinstance(enabled, list) or not all(isinstance(v, str) for v in enabled):
            raise ConfigError("rules.enabled: expected a list of strings")
        rules.enabled = tuple(enabled)

    if "cameras" in raw:
        cams_raw = raw["cameras"]
        if not isinstance(cams_raw, list):
            raise ConfigError("cameras: expected an array of tables")
        config.cameras = [_load_camera(cam, i) for i, cam in enumerate(cams_raw)]

    return config.validate()


def resolve_path(base: str | Path, value: str) -> Path:
    """Resolve a possibly relative path against the project root."""
    p = Path(value)
    return p if p.is_absolute() else Path(base) / p
