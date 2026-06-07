import os
import pytest
from src.neo_features import FEATURE_COLUMNS

pytestmark = pytest.mark.skipif(
    not os.path.exists("model.joblib"),
    reason="model.joblib not built yet (run the notebook task first)",
)


def test_predict_returns_label_and_probability():
    from app import predict
    out = predict(20.0, 0.02, 0.1, 15.0, 3)
    assert isinstance(out, str)
    assert "%" in out  # probability is shown


def test_model_features_match_contract():
    import joblib
    bundle = joblib.load("model.joblib")
    assert bundle["features"] == FEATURE_COLUMNS
