"""Route low-confidence predictions through the context agent."""

from typing import Literal

from langgraph.types import Command

from masar.nodes import predictor as predictor_module
from masar.state import MasarState


def router_node(state: MasarState) -> Command[Literal["context_agent", "route_optimizer"]]:
    confidence = state.get("prediction_confidence")
    is_anomaly = confidence is None or confidence < predictor_module.CONFIDENCE_THRESHOLD
    return Command(
        update={
            "is_anomaly": is_anomaly,
            "route_path": "deep" if is_anomaly else "fast",
        },
        goto="context_agent" if is_anomaly else "route_optimizer",
    )