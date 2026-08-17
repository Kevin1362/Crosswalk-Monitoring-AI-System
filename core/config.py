from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in (None, ""):
        return float(default)
    return float(value)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in (None, ""):
        return int(default)
    return int(value)


def load_settings(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle) or {}

    # Override selected GuideKaro configuration using environment variables.
    settings.setdefault("detection", {})
    settings["detection"]["confidence"] = _float_env(
        "MODEL_CONFIDENCE_THRESHOLD",
        settings["detection"].get("confidence", 0.35),
    )

    settings.setdefault("tracking", {})
    settings["tracking"]["pixels_per_meter"] = _float_env(
        "PIXELS_PER_METER",
        settings["tracking"].get("pixels_per_meter", 22.0),
    )

    # Assignment experiment configuration.
    feature_names = os.getenv(
        "FEATURE_NAMES",
        "risk_score,vehicle_speed_kmh,distance_to_crosswalk_m,"
        "pedestrian_vehicle_distance_m,confidence,ttc_seconds",
    )

    settings["experiment"] = {
        "feature_names": [
            item.strip()
            for item in feature_names.split(",")
            if item.strip()
        ],
        "expected_accuracy": _float_env("EXPECTED_ACCURACY", 0.90),
        "num_epochs": _int_env("NUM_EPOCHS", 50),
        "name": os.getenv("EXPERIMENT_NAME", "guidekaro_enhanced"),
        "version": os.getenv("EXPERIMENT_VERSION", "1.0"),
    }

    # Workshop DB variables.
    # GuideKaro itself currently uses SQLite.
    settings["database_environment"] = {
        "user": os.getenv("DB_USER", ""),
        "password": os.getenv("DB_PASSWORD", ""),
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
    }

    return settings