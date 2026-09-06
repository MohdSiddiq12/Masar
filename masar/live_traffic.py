"""Bridge live Supabase ``traffic_logs`` rows into MasarState fields."""

import json
from datetime import datetime, timezone
from typing import Any, Optional

LOCATION_TO_TRAFFIC_KEY = {
    "Sheikh Zayed Road": "Sheikh_Zayed_Rd",
    "Al Khail": "Al_Khail_Rd",
    "Business Bay": "Business_Bay",
    "Marina": "Dubai_Marina",
    "Airport": "Airport_Area",
}

TRAFFIC_COLUMNS = "current_speed,free_flow_speed,speed_ratio,weather_main,rain_mm,raw_data,created_at"


def fetch_latest_row(client: Any, location: str) -> Optional[dict]:
    traffic_key = LOCATION_TO_TRAFFIC_KEY.get(location)
    if traffic_key is None:
        return None

    from masar.api_report import measured_call

    response = measured_call(
        "supabase",
        "traffic_logs.latest",
        {"table": "traffic_logs", "location": traffic_key, "select": TRAFFIC_COLUMNS, "order": "created_at desc", "limit": 1},
        lambda: (
            client.table("traffic_logs")
            .select(TRAFFIC_COLUMNS)
            .eq("location_name", traffic_key)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        ),
    )
    rows = response.data or []
    return rows[0] if rows else None


def _row_temperature(row: dict) -> Optional[float]:
    raw_data = row.get("raw_data") or {}
    if isinstance(raw_data, str):
        raw_data = json.loads(raw_data)
    weather = raw_data.get("weather") or {}
    temp = (weather.get("main") or {}).get("temp")
    return float(temp) if temp is not None else None


def row_to_state_fields(row: dict) -> dict:
    weather_main = row.get("weather_main") or "Clear"
    rain_mm = row.get("rain_mm") or 0
    fields = {
        "current_speed": float(row["current_speed"]),
        "free_flow_speed": float(row["free_flow_speed"]),
        "congestion_ratio": float(row["speed_ratio"]),
        "weather_condition": weather_main,
        "is_raining": bool(rain_mm) or "rain" in weather_main.lower(),
    }
    temperature = _row_temperature(row)
    if temperature is not None:
        fields["temperature"] = temperature
    return fields


def row_age_seconds(row: dict, now: Optional[datetime] = None) -> Optional[float]:
    created_at = row.get("created_at")
    if not created_at:
        return None
    now = now or datetime.now(timezone.utc)
    try:
        created = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (now - created).total_seconds()


def get_live_fields(client: Any, location: str) -> tuple[dict, Optional[dict]]:
    row = fetch_latest_row(client, location)
    if row is None:
        return {}, None
    return row_to_state_fields(row), row