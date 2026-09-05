"""Shared LangGraph state contract."""

import operator
from typing import Annotated, Literal, Optional, TypedDict


class MasarState(TypedDict):
    location: str
    timestamp: str
    current_speed: float
    free_flow_speed: float
    congestion_ratio: float
    incidents: Annotated[list[str], operator.add]
    temperature: float
    weather_condition: str
    is_raining: bool
    predicted_congestion: Optional[float]
    prediction_confidence: Optional[float]
    is_anomaly: bool
    route_path: Literal["fast", "deep"]
    context_notes: Optional[str]
    nearby_events: Annotated[list[str], operator.add]
    social_signal: Optional[str]
    recommended_mode: Optional[Literal["drive", "metro", "drive_to_metro"]]
    recommended_route: Annotated[list[str], operator.add]
    message_en: Optional[str]
    message_ar: Optional[str]