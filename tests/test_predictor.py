import pytest

import masar.nodes.predictor as predictor_module
from masar.nodes.predictor import predictor_node, CONFIDENCE_THRESHOLD


@pytest.fixture(autouse=True)
def no_trained_model(monkeypatch, tmp_path):
    monkeypatch.setattr(predictor_module, "MODEL_PATH", str(tmp_path / "missing.pkl"))
    monkeypatch.setattr(predictor_module, "_model", None)
    monkeypatch.setattr(predictor_module, "_model_checked", False)


def make_state(**overrides) -> dict:
    """A minimal valid MasarState for testing a single node in isolation."""
    base = {
        "location": "Sheikh Zayed Road",
        "timestamp": "2026-09-05T18:00:00Z",
        "current_speed": 40.0,
        "free_flow_speed": 100.0,
        "congestion_ratio": 0.4,
        "incidents": [],
        "temperature": 38.0,
        "weather_condition": "Clear",
        "is_raining": False,
        "predicted_congestion": None,
        "prediction_confidence": None,
        "is_anomaly": False,
        "route_path": "fast",
        "context_notes": None,
        "nearby_events": [],
        "social_signal": None,
        "recommended_mode": None,
        "recommended_route": [],
        "message_en": None,
        "message_ar": None,
    }
    base.update(overrides)
    return base


def test_predictor_returns_required_keys():
    result = predictor_node(make_state())
    assert "predicted_congestion" in result
    assert "prediction_confidence" in result


def test_predictor_values_are_valid_probabilities():
    result = predictor_node(make_state(congestion_ratio=0.3))
    assert 0.0 <= result["predicted_congestion"] <= 1.0
    assert 0.0 <= result["prediction_confidence"] <= 1.0


def test_no_trained_model_falls_back_to_low_confidence_heuristic():
    # No model file exists in this test environment -> heuristic path.
    # Confidence must stay below threshold so the Router sends this to Context.
    result = predictor_node(make_state())
    assert result["prediction_confidence"] < CONFIDENCE_THRESHOLD


def test_heavier_congestion_yields_higher_predicted_score():
    light = predictor_node(make_state(congestion_ratio=0.9))  # near free-flow speed
    heavy = predictor_node(make_state(congestion_ratio=0.2))  # heavy congestion
    assert heavy["predicted_congestion"] > light["predicted_congestion"]


def test_does_not_mutate_input_state():
    state = make_state()
    original = dict(state)
    predictor_node(state)
    assert state == original
