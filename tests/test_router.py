from langgraph.types import Command

from masar.nodes.router import router_node


def test_low_confidence_routes_to_context(make_state):
    result = router_node(make_state(prediction_confidence=0.4))
    assert isinstance(result, Command)
    assert result.goto == "context_agent"
    assert result.update == {"is_anomaly": True, "route_path": "deep"}


def test_missing_confidence_fails_safe_to_context(make_state):
    result = router_node(make_state(prediction_confidence=None))
    assert result.goto == "context_agent"


def test_high_confidence_routes_to_optimizer(make_state):
    result = router_node(make_state(prediction_confidence=0.9))
    assert result.goto == "route_optimizer"
    assert result.update["route_path"] == "fast"