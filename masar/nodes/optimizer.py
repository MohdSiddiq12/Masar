"""Route optimizer for the monitored Dubai corridors."""

import networkx as nx

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
    "Marina": (25.0800, 55.1400),
    "Sheikh Zayed Road": (25.2050, 55.2700),
    "Business Bay": (25.1850, 55.2650),
    "Al Khail": (25.1500, 55.2400),
    "Airport": (25.2532, 55.3657),
}

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


def _recommend_mode(congestion: float) -> str:
    if congestion > 0.6:
        return "metro"
    if congestion > 0.3:
        return "drive_to_metro"
    return "drive"


def route_optimizer_node(state: MasarState) -> dict:
    congestion = state.get("predicted_congestion") or 0.0
    mode = _recommend_mode(congestion)
    if mode == "metro":
        return {
            "recommended_route": [],
            "recommended_mode": mode,
            "route_options": [],
        }

    base_graph = _build_base_graph()
    graph = base_graph.copy()
    for source, target in graph.edges():
        sensitivity = _CONGESTION_SENSITIVITY[frozenset((source, target))]
        penalty = congestion * sensitivity if congestion > 0.3 else congestion
        graph[source][target]["weight"] *= 1 + penalty

    origin = state.get("origin") or "Marina"
    destination = state.get("destination") or "Business Bay"
    try:
        path = nx.dijkstra_path(graph, origin, destination, weight="weight")
        baseline_path = nx.dijkstra_path(base_graph, origin, destination, weight="weight")
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
                "estimated_minutes": round(_path_weight(graph if label == "Recommended" else base_graph, route_path), 1),
                "congestion_percent": round(_path_congestion(route_path, congestion) * 100),
                "coordinates": [list(LOCATION_COORDINATES[location]) for location in route_path],
                "recommended": label == "Recommended",
            }
        )

    return {
        "recommended_route": path,
        "recommended_mode": mode,
        "route_options": route_options,
    }


def _path_weight(graph: nx.Graph, path: list[str]) -> float:
    return sum(graph[source][target]["weight"] for source, target in zip(path, path[1:]))


def _path_congestion(path: list[str], congestion: float) -> float:
    if not path or len(path) < 2:
        return congestion
    sensitivities = [
        _CONGESTION_SENSITIVITY[frozenset((source, target))]
        for source, target in zip(path, path[1:])
    ]
    return min(0.99, congestion * (sum(sensitivities) / len(sensitivities)))