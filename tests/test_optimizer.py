from masar.nodes.optimizer import route_optimizer_node


def test_optimizer_returns_route_and_mode(make_state):
    result = route_optimizer_node(make_state(predicted_congestion=0.2))
    assert result["recommended_route"] == ["Marina", "Al Khail", "Business Bay"]
    assert result["recommended_mode"] == "drive"


def test_optimizer_switches_mode_for_heavy_congestion(make_state):
    result = route_optimizer_node(make_state(predicted_congestion=0.8))
    assert result["recommended_mode"] == "metro"


def test_optimizer_handles_unknown_endpoints(make_state):
    result = route_optimizer_node(make_state(origin="Unknown", destination="Business Bay"))
    assert result["recommended_route"] == []