"""Compile the Masar predictor-to-synthesis workflow."""

from langgraph.graph import END, START, StateGraph

from masar.nodes.context import context_node
from masar.nodes.optimizer import route_optimizer_node
from masar.nodes.predictor import predictor_node
from masar.nodes.router import router_node
from masar.nodes.synthesis import synthesis_node
from masar.state import MasarState


def build_graph(context_llm=None, synthesis_llm=None):
    graph = StateGraph(MasarState)
    graph.add_node("predictor", predictor_node)
    graph.add_node("router", router_node)
    graph.add_node("context_agent", lambda state: context_node(state, llm=context_llm))
    graph.add_node("route_optimizer", route_optimizer_node)
    graph.add_node("synthesis", lambda state: synthesis_node(state, llm=synthesis_llm))

    graph.add_edge(START, "predictor")
    graph.add_edge("predictor", "router")
    graph.add_edge("context_agent", "route_optimizer")
    graph.add_edge("route_optimizer", "synthesis")
    graph.add_edge("synthesis", END)
    return graph.compile()