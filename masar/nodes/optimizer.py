"""Route optimizer for Dubai locations."""

import networkx as nx

from masar.locations import LOCATIONS
from masar.state import MasarState

_BASE_EDGES = [
    ("Sheikh Zayed Road", "Business Bay", 8),
    ("Sheikh Zayed Road", "Marina", 15),
    ("Business Bay", "Al Khail", 10),
    ("Al Khail", "Marina", 12),
    ("Al Khail", "Airport", 20),
    ("Business Bay", "Airport", 18),
]

LOCATION_COORDINATES = {
    location["label"]: (location["lat"], location["lon"])
    for location in LOCATIONS
}
_LEGACY_LOCATIONS = {location for edge in _BASE_EDGES for location in edge[:2]}

# Relative corridor sensitivity used when traffic is heavy. The values are
# deliberately explainable and can later be replaced by per-road telemetry.
_CONGESTION_SENSITIVITY = {
    frozenset(("Sheikh Zayed Road", "Business Bay")): 0.70,
    frozenset(("Sheikh Zayed Road", "Marina")): 0.75,
    frozenset(("Business Bay", "Al Khail")): 1.35,
    frozenset(("Al Khail", "Marina")): 1.55,
    frozenset(("Al Khail", "Airport")): 1.15,
    frozenset(("Business Bay", "Airport")): 1.00,
}


def _build_base_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_weighted_edges_from(_BASE_EDGES)
    return graph


def _build_location_graph() -> nx.Graph:
    graph = _build_base_graph()
    for location in LOCATION_COORDINATES:
        if location in _LEGACY_LOCATIONS:
            continue
        nearest_hubs = sorted(
            _LEGACY_LOCATIONS,
            key=lambda hub: _distance_minutes(location, hub),
        )[:2]
        for hub in nearest_hubs:
            graph.add_edge(location, hub, weight=_distance_minutes(location, hub))
    return graph


def _distance_minutes(source: str, target: str) -> float:
    source_lat, source_lon = LOCATION_COORDINATES[source]
    target_lat, target_lon = LOCATION_COORDINATES[target]
    latitude_km = (source_lat - target_lat) * 111
    longitude_km = (source_lon - target_lon) * 100
    return max(3.0, round((latitude_km**2 + longitude_km**2) ** 0.5 / 0.65, 1))


def _recommend_mode(congestion: float) -> str:
    if congestion > 0.6:
        return "metro"
    if congestion > 0.3:
        return "drive_to_metro"
    return "drive"


def route_optimizer_node(state: MasarState) -> dict:
    congestion = state.get("predicted_congestion") or 0.0
    mode = _recommend_mode(congestion)

    origin = state.get("origin") or "Marina"
    destination = state.get("destination") or "Business Bay"
    base_graph = _build_base_graph()
    uses_expanded_graph = not (
        origin in _LEGACY_LOCATIONS and destination in _LEGACY_LOCATIONS
    )
    if not uses_expanded_graph:
        graph = base_graph.copy()
    else:
        graph = _build_location_graph()
    baseline_graph = _build_location_graph() if uses_expanded_graph else base_graph
    for source, target in graph.edges():
        sensitivity = _CONGESTION_SENSITIVITY.get(frozenset((source, target)), 1.0)
        penalty = congestion * sensitivity if congestion > 0.3 else congestion
        graph[source][target]["weight"] *= 1 + penalty

    try:
        path = nx.dijkstra_path(graph, origin, destination, weight="weight")
        baseline_path = nx.dijkstra_path(baseline_graph, origin, destination, weight="weight")
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        path = []
        baseline_path = []

    route_options = []
    for label, route_path in (("Recommended", path), ("Fastest baseline", baseline_path)):
        if not route_path:
            continue
        route_options.append(
            {
                "label": label,
                "path": route_path,
                "estimated_minutes": round(_path_weight(graph if label == "Recommended" else baseline_graph, route_path), 1),
                "congestion_percent": round(_path_congestion(route_path, congestion) * 100),
                "coordinates": [list(LOCATION_COORDINATES[location]) for location in route_path],
                "recommended": label == "Recommended",
            }
        )

    return {
        # Metro remains the primary recommendation for severe congestion, but
        # keep the lower-congestion road alternative visible and actionable.
        "recommended_route": [] if mode == "metro" else path,
        "recommended_mode": mode,
        "route_options": route_options,
    }


def _path_weight(graph: nx.Graph, path: list[str]) -> float:
    return sum(graph[source][target]["weight"] for source, target in zip(path, path[1:]))


def _path_congestion(path: list[str], congestion: float) -> float:
    if not path or len(path) < 2:
        return congestion
    sensitivities = [
        _CONGESTION_SENSITIVITY.get(frozenset((source, target)), 1.0)
        for source, target in zip(path, path[1:])
    ]
    return min(0.99, congestion * (sum(sensitivities) / len(sensitivities)))