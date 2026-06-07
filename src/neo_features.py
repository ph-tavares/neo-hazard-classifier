"""Pure, reusable logic shared by collection, notebook, and app.

Feature engineering for the NEO hazard classifier:
- One row per asteroid.
- Approach features aggregated over Earth close approaches only
  (the PHA flag is defined relative to Earth).
- Diameter columns are emitted for EDA (to show their collinearity with
  absolute_magnitude_h) but are NOT part of FEATURE_COLUMNS.
"""
from __future__ import annotations

import pandas as pd

EARTH = "Earth"

# Columns actually fed to the models (single size proxy + engineered approach features).
FEATURE_COLUMNS = [
    "absolute_magnitude_h",
    "miss_distance_min_au",
    "miss_distance_mean_au",
    "relative_velocity_max_kms",
    "n_close_approaches",
]

TARGET_COLUMN = "is_potentially_hazardous_asteroid"


def extract_features(neo: dict) -> dict | None:
    """Turn one NeoWs `browse` object into a flat feature row.

    Returns None when the object cannot be used (no Earth approach, or no
    absolute magnitude), so the caller can simply skip it.
    """
    approaches = [
        c for c in (neo.get("close_approach_data") or [])
        if c.get("orbiting_body") == EARTH
    ]
    if not approaches:
        return None

    h = neo.get("absolute_magnitude_h")
    if h is None:
        return None

    miss_au = [float(c["miss_distance"]["astronomical"]) for c in approaches]
    vel_kms = [float(c["relative_velocity"]["kilometers_per_second"]) for c in approaches]

    diameter_km = (neo.get("estimated_diameter") or {}).get("kilometers") or {}

    return {
        "designation": neo.get("designation"),
        "name": neo.get("name"),
        "absolute_magnitude_h": float(h),
        "estimated_diameter_min_km": _maybe_float(diameter_km.get("estimated_diameter_min")),
        "estimated_diameter_max_km": _maybe_float(diameter_km.get("estimated_diameter_max")),
        "miss_distance_min_au": min(miss_au),
        "miss_distance_mean_au": sum(miss_au) / len(miss_au),
        "relative_velocity_max_kms": max(vel_kms),
        "n_close_approaches": len(approaches),
        TARGET_COLUMN: int(bool(neo.get("is_potentially_hazardous_asteroid"))),
    }


def build_input_frame(values: dict) -> pd.DataFrame:
    """Build a single-row DataFrame with columns in FEATURE_COLUMNS order.

    Used by the app so the model always sees features in the training order.
    """
    return pd.DataFrame([{c: values[c] for c in FEATURE_COLUMNS}], columns=FEATURE_COLUMNS)


def _maybe_float(value):
    return None if value is None else float(value)
