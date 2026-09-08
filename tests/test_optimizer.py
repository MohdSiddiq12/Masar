from masar.nodes.optimizer import route_optimizer_node


def test_optimizer_returns_route_and_mode(make_state):
    result = route_optimizer_node(make_state(predicted_congestion=0.2))
    assert result["recommended_route"] == ["Marina", "Al Khail", "Business Bay"]
    assert result["recommended_mode"] == "drive"


def test_optimizer_switches_mode_for_heavy_congestion(make_state):
    result = route_optimizer_node(make_state(predicted_congestion=0.8))
    assert result["recommended_mode"] == "metro"
    assert result["recommended_route"] == []
    assert result["route_options"][0]["recommended"] is True
    assert result["route_options"][0]["coordinates"]


def test_optimizer_avoids_congested_al_khail_corridor(make_state):
    result = route_optimizer_node(make_state(predicted_congestion=0.5))
    assert result["recommended_route"] == ["Marina", "Sheikh Zayed Road", "Business Bay"]
    assert result["recommended_mode"] == "drive_to_metro"
    assert result["route_options"][0]["recommended"] is True
    assert result["route_options"][0]["coordinates"][0] == [25.08, 55.14]


def test_optimizer_exposes_baseline_route_for_comparison(make_state):
    result = route_optimizer_node(make_state(predicted_congestion=0.2))
    assert [option["label"] for option in result["route_options"]] == [
        "Recommended",
        "Fastest baseline",
    ]
    assert result["route_options"][0]["path"] == result["route_options"][1]["path"]


def test_optimizer_handles_unknown_endpoints(make_state):
    result = route_optimizer_node(make_state(origin="Unknown", destination="Business Bay"))
    assert result["recommended_route"] == []


def test_optimizer_routes_new_dubai_locations(make_state):
    result = route_optimizer_node(
        make_state(origin="Deira", destination="Marina")
    )
    assert result["route_options"]
    assert result["route_options"][0]["path"][0] == "Deira"
    assert result["route_options"][0]["path"][-1] == "Marina"
    assert result["route_options"][0]["coordinates"]