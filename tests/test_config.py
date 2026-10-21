from __future__ import annotations

import pytest

from visioniq.config import (
    ConfigError,
    DEFAULT_CAMERAS,
    ModelConfig,
    Rect,
    default_config,
    load_config,
)


def test_default_deployment_matches_mockup():
    config = default_config()
    assert len(config.cameras) == 8
    assert config.cameras[0].id == "CAM-01"
    assert config.cameras[0].name == "Main Entrance"
    assert config.cameras[0].restricted
    assert config.cameras[1].source == "synthetic:parking"
    assert {c.id for c in config.cameras} == {c.id for c in DEFAULT_CAMERAS}


def test_rect_and_after_hours():
    rect = Rect.parse([0.1, 0.2, 0.3, 0.4])
    assert rect.contains(0.2, 0.3)
    assert not rect.contains(0.9, 0.9)
    with pytest.raises(ConfigError):
        Rect.parse([0.9, 0.9, 0.2, 0.2])
    with pytest.raises(ConfigError):
        Rect.parse([0.0, 0.0, -1.0, 0.2])

    cam = default_config().cameras[0]
    assert cam.is_after_hours(23.0)
    assert cam.is_after_hours(2.0)
    assert not cam.is_after_hours(12.0)


def test_model_config_validation():
    with pytest.raises(ConfigError):
        ModelConfig(backend="magic")
    with pytest.raises(ConfigError):
        ModelConfig(window=2)
    with pytest.raises(ConfigError):
        ModelConfig(input_clip=0.5)
    with pytest.raises(ConfigError):
        ModelConfig(threshold_percentile=100.0)


def test_toml_roundtrip_and_unknown_keys(tmp_path):
    path = tmp_path / "visioniq.toml"
    path.write_text(
        """
[app]
db_path = "state/events.db"
display_width = 320

[model]
window = 12
warmup_windows = 40
backend = "numpy"

[rules]
enabled = ["anomaly", "crowding"]

[[cameras]]
id = "CAM-01"
name = "Test Gate"
zone = "Test / Zone"
source = "synthetic:lobby"
restricted = true
access_controlled = true
after_hours = [22, 6]
zones = [{ name = "door", kind = "entry", rect = [0.7, 0.3, 0.2, 0.5] }]
""",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.db_path == "state/events.db"
    assert config.display_width == 320
    assert config.model.window == 12
    assert config.model.backend == "numpy"
    assert config.rules.enabled == ("anomaly", "crowding")
    assert len(config.cameras) == 1
    assert config.cameras[0].after_hours == (22.0, 6.0)
    assert config.cameras[0].entry_zone is not None

    path.write_text("[model]\nnot_a_key = 3\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(path)


def test_missing_config_and_duplicate_ids(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")

    path = tmp_path / "dup.toml"
    path.write_text(
        """
[[cameras]]
id = "CAM-01"
name = "A"
[[cameras]]
id = "CAM-01"
name = "B"
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="duplicate"):
        load_config(path)
