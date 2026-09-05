import pytest


@pytest.fixture
def make_state():
    def factory(**overrides):
        state = {
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
            "origin": "Marina",
            "destination": "Business Bay",
            "context_notes": None,
            "nearby_events": [],
            "social_signal": None,
            "recommended_mode": None,
            "recommended_route": [],
            "message_en": None,
            "message_ar": None,
        }
        state.update(overrides)
        return state

    return factory