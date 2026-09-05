"""Interim route optimizer for the monitored Dubai corridors."""

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
    graph = _build_base_graph()
    congestion = state.get("predicted_congestion") or 0.0
    for source, target in graph.edges():
        graph[source][target]["weight"] *= 1 + congestion

    origin = state.get("origin") or "Marina"
    destination = state.get("destination") or "Business Bay"
    try:
        path = nx.dijkstra_path(graph, origin, destination, weight="weight")
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        path = []

    return {
        "recommended_route": path,
        "recommended_mode": _recommend_mode(congestion),
    }