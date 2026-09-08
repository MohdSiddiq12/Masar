"""Fetch and optionally persist the live traffic and weather signal."""

import asyncio
import os
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from masar.api_report import measured_call, record_call


LOCATIONS = [
    {"name": "Sheikh_Zayed_Rd", "label": "Sheikh Zayed Road", "lat": 25.2048, "lon": 55.2708},
    {"name": "Al_Khail_Rd", "label": "Al Khail", "lat": 25.1200, "lon": 55.2400},
    {"name": "Business_Bay", "label": "Business Bay", "lat": 25.1850, "lon": 55.2650},
    {"name": "Dubai_Marina", "label": "Marina", "lat": 25.0800, "lon": 55.1400},
    {"name": "Airport_Area", "label": "Airport", "lat": 25.2532, "lon": 55.3657},
]
LOCATION_BY_LABEL = {location["label"]: location for location in LOCATIONS}


async def _fetch_tomtom(client: httpx.AsyncClient, lat: float, lon: float, api_key: str) -> dict:
    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/12/json"
    params = {"key": api_key, "point": f"{lat},{lon}", "unit": "KMPH"}
    started = time.perf_counter()
    try:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    except Exception as error:
        record_call("tomtom", "traffic_flow", {"url": url, "params": {"point": params["point"], "unit": "KMPH"}}, status="error", duration_ms=(time.perf_counter() - started) * 1000, error=error)
        raise
    record_call("tomtom", "traffic_flow", {"url": url, "params": {"point": params["point"], "unit": "KMPH"}}, data, duration_ms=(time.perf_counter() - started) * 1000)
    flow = data["flowSegmentData"]
    current = flow["currentSpeed"]
    free = flow["freeFlowSpeed"]
    return {
        "current_speed": current,
        "free_flow_speed": free,
        "speed_ratio": round(current / free, 3) if free else None,
        "delay_seconds": flow["currentTravelTime"] - flow["freeFlowTravelTime"],
        "raw": flow,
    }


async def _fetch_weather(client: httpx.AsyncClient, lat: float, lon: float, api_key: str) -> dict:
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
    started = time.perf_counter()
    try:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    except Exception as error:
        record_call("openweathermap", "current_weather", {"url": url, "params": {"lat": lat, "lon": lon, "units": "metric"}}, status="error", duration_ms=(time.perf_counter() - started) * 1000, error=error)
        raise
    record_call("openweathermap", "current_weather", {"url": url, "params": {"lat": lat, "lon": lon, "units": "metric"}}, data, duration_ms=(time.perf_counter() - started) * 1000)
    rain = data.get("rain", {}).get("1h", 0) or data.get("rain", {}).get("3h", 0) or 0
    return {
        "weather_main": data["weather"][0]["main"],
        "description": data["weather"][0]["description"],
        "temp": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "rain_mm": rain,
        "raw": data,
    }


async def fetch_live_row(location: str) -> dict:
    selected = LOCATION_BY_LABEL.get(location)
    if selected is None:
        raise ValueError(f"Unsupported live-data location: {location}")
    tomtom_key = os.getenv("TOMTOM_API_KEY")
    weather_key = os.getenv("OPENWEATHER_API_KEY")
    if not tomtom_key or not weather_key:
        raise RuntimeError("TOMTOM_API_KEY and OPENWEATHER_API_KEY are required for live recommendations")

    async with httpx.AsyncClient(timeout=10) as client:
        traffic, weather = await asyncio.gather(
            _fetch_tomtom(client, selected["lat"], selected["lon"], tomtom_key),
            _fetch_weather(client, selected["lat"], selected["lon"], weather_key),
        )
    return {
        "location_name": selected["name"],
        "lat": selected["lat"],
        "lon": selected["lon"],
        "current_speed": traffic["current_speed"],
        "free_flow_speed": traffic["free_flow_speed"],
        "speed_ratio": traffic["speed_ratio"],
        "delay_seconds": traffic["delay_seconds"],
        "weather_main": weather["weather_main"],
        "rain_mm": weather["rain_mm"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_data": {"tomtom": traffic["raw"], "weather": weather["raw"]},
    }


async def fetch_and_store(location: str, supabase_client: Any) -> dict:
    row = await fetch_live_row(location)
    measured_call(
        "supabase",
        "traffic_logs.insert",
        {"table": "traffic_logs", "row": row},
        lambda: supabase_client.table("traffic_logs").insert(row).execute(),
    )
    return row
