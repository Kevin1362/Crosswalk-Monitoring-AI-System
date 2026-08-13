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


def _secret_or_env(name: str, default: str = "") -> str:
    """
    Read a sensitive value from a Docker Secret first.

    Example:
        DB_PASSWORD_FILE=/run/secrets/db_password

    If a Docker Secret is not available, fall back to the
    normal environment variable.
    """
    secret_file = os.getenv(f"{name}_FILE")

    if secret_file:
        secret_path = Path(secret_file)

        if secret_path.is_file():
            try:
                return secret_path.read_text(
                    encoding="utf-8"
                ).strip()
            except OSError:
                pass

    return os.getenv(name, default)


def load_settings(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle) or {}

    # --------------------------------------------------
    # GuideKaro detection configuration
    # --------------------------------------------------

    settings.setdefault("detection", {})

    settings["detection"]["confidence"] = _float_env(
        "MODEL_CONFIDENCE_THRESHOLD",
        settings["detection"].get("confidence", 0.35),
    )

    # --------------------------------------------------
    # Tracking configuration
    # --------------------------------------------------

    settings.setdefault("tracking", {})

    settings["tracking"]["pixels_per_meter"] = _float_env(
        "PIXELS_PER_METER",
        settings["tracking"].get("pixels_per_meter", 22.0),
    )

    # --------------------------------------------------
    # EDA and experiment configuration
    # --------------------------------------------------

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
        "expected_accuracy": _float_env(
            "EXPECTED_ACCURACY",
            0.90,
        ),
        "num_epochs": _int_env(
            "NUM_EPOCHS",
            50,
        ),
        "name": os.getenv(
            "EXPERIMENT_NAME",
            "guidekaro_enhanced",
        ),
        "version": os.getenv(
            "EXPERIMENT_VERSION",
            "1.0",
        ),
    }

    # --------------------------------------------------
    # Database / Secrets Management configuration
    # --------------------------------------------------
    # GuideKaro currently uses SQLite.
    # These variables demonstrate secure configuration
    # management required by the DevOps assignment.
    #
    # DB_PASSWORD is loaded from Docker Secret when
    # DB_PASSWORD_FILE is available.
    # --------------------------------------------------

    settings["database_environment"] = {
        "user": os.getenv(
            "DB_USER",
            "",
        ),
        "password": _secret_or_env(
            "DB_PASSWORD",
            "",
        ),
        "host": os.getenv(
            "DB_HOST",
            "localhost",
        ),
        "port": os.getenv(
            "DB_PORT",
            "5432",
        ),
    }

    return settings