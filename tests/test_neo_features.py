import math
import pandas as pd
from src.neo_features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    extract_features,
    build_input_frame,
)


def _neo(cad, h=20.0, pha=True):
    """Minimal NeoWs-shaped object (fields mirror the live API)."""
    return {
        "designation": "1620",
        "name": "1620 Geographos",
        "absolute_magnitude_h": h,
        "estimated_diameter": {
            "kilometers": {
                "estimated_diameter_min": 2.3472263753,
                "estimated_diameter_max": 5.2485577338,
            }
        },
        "is_potentially_hazardous_asteroid": pha,
        "close_approach_data": cad,
    }


def _approach(km_s, au, body="Earth"):
    return {
        "relative_velocity": {"kilometers_per_second": str(km_s)},
        "miss_distance": {"astronomical": str(au)},
        "orbiting_body": body,
    }


def test_extract_features_aggregates_earth_approaches():
    neo = _neo([_approach(10.0, 0.05), _approach(20.0, 0.01)], h=15.27, pha=True)
    f = extract_features(neo)
    assert f["absolute_magnitude_h"] == 15.27
    assert f["relative_velocity_max_kms"] == 20.0
    assert f["miss_distance_min_au"] == 0.01
    assert math.isclose(f["miss_distance_mean_au"], 0.03)
    assert f["n_close_approaches"] == 2
    assert f["is_potentially_hazardous_asteroid"] == 1
    assert f["estimated_diameter_min_km"] == 2.3472263753
    assert f["estimated_diameter_max_km"] == 5.2485577338


def test_extract_features_ignores_non_earth_bodies():
    neo = _neo([_approach(10.0, 0.05, body="Venus"), _approach(20.0, 0.01, body="Earth")])
    f = extract_features(neo)
    assert f["n_close_approaches"] == 1
    assert f["relative_velocity_max_kms"] == 20.0


def test_extract_features_none_without_earth_approach():
    neo = _neo([_approach(10.0, 0.05, body="Venus")])
    assert extract_features(neo) is None


def test_extract_features_none_without_magnitude():
    neo = _neo([_approach(10.0, 0.05)], h=None)
    assert extract_features(neo) is None


def test_extract_features_none_with_empty_cad():
    assert extract_features(_neo([])) is None


def test_target_label_is_zero_for_non_pha():
    f = extract_features(_neo([_approach(5.0, 0.2)], pha=False))
    assert f["is_potentially_hazardous_asteroid"] == 0


def test_constants_are_consistent():
    assert TARGET_COLUMN == "is_potentially_hazardous_asteroid"
    assert FEATURE_COLUMNS == [
        "absolute_magnitude_h",
        "miss_distance_min_au",
        "miss_distance_mean_au",
        "relative_velocity_max_kms",
        "n_close_approaches",
    ]


def test_build_input_frame_orders_columns():
    values = {
        "absolute_magnitude_h": 21.0,
        "miss_distance_min_au": 0.02,
        "miss_distance_mean_au": 0.1,
        "relative_velocity_max_kms": 15.0,
        "n_close_approaches": 3,
    }
    frame = build_input_frame(values)
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == FEATURE_COLUMNS
    assert frame.shape == (1, 5)
    assert frame.iloc[0]["absolute_magnitude_h"] == 21.0
