"""Shared feature definitions and transformations for congestion models."""

import json
from typing import Any

import numpy as np

FEATURE_ORDER = [
    "current_speed",
    "free_flow_speed",
    "congestion_ratio",
    "temperature",
    "is_raining",
]


def _weather_temperature(row: dict[str, Any]) -> float:
    raw_data = row.get("raw_data") or {}
    if isinstance(raw_data, str):
        raw_data = json.loads(raw_data)
    weather = raw_data.get("weather") or {}
    return float((weather.get("main") or {}).get("temp", row.get("temperature", 30.0)))


def row_to_features(row: dict[str, Any]) -> list[float]:
    """Convert a traffic_logs row or a normalized row to FEATURE_ORDER."""
    ratio = row.get("congestion_ratio", row.get("speed_ratio"))
    rain_mm = row.get("rain_mm", 0) or 0
    weather_main = str(row.get("weather_main", "")).lower()
    return [
        float(row["current_speed"]),
        float(row["free_flow_speed"]),
        float(ratio),
        _weather_temperature(row),
        float(bool(rain_mm) or "rain" in weather_main),
    ]


def features_to_array(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([row_to_features(row) for row in rows], dtype=float)