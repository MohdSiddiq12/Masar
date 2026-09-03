import os
import httpx
from datetime import datetime, timezone
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# === Clients ===
TOMTOM_KEY = os.getenv("TOMTOM_API_KEY")
WEATHER_KEY = os.getenv("OPENWEATHER_API_KEY")
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# === Key locations in Dubai ===
LOCATIONS = [
    {"name": "Sheikh_Zayed_Rd", "lat": 25.2048, "lon": 55.2708},
    {"name": "Al_Khail_Rd",     "lat": 25.1200, "lon": 55.2400},
    {"name": "Business_Bay",    "lat": 25.1850, "lon": 55.2650},
    {"name": "Dubai_Marina",    "lat": 25.0800, "lon": 55.1400},
    {"name": "Airport_Area",    "lat": 25.2532, "lon": 55.3657},
]

async def get_tomtom_flow(lat: float, lon: float) -> dict:
    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/12/json"
    params = {"key": TOMTOM_KEY, "point": f"{lat},{lon}", "unit": "KMPH"}
    
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()["flowSegmentData"]
        
        current = data["currentSpeed"]
        free = data["freeFlowSpeed"]
        return {
            "current_speed": current,
            "free_flow_speed": free,
            "speed_ratio": round(current / free, 3) if free else None,
            "delay_seconds": data["currentTravelTime"] - data["freeFlowTravelTime"],
            "confidence": data.get("confidence"),
            "road_closure": data.get("roadClosure", False),
            "raw": data
        }

async def get_weather(lat: float, lon: float) -> dict:
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": WEATHER_KEY, "units": "metric"}
    
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        
        rain = data.get("rain", {}).get("1h", 0) or data.get("rain", {}).get("3h", 0) or 0
        return {
            "weather_main": data["weather"][0]["main"],
            "description": data["weather"][0]["description"],
            "temp": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "rain_mm": rain,
            "raw": data
        }

async def collect_and_store():
    print(f"Starting collection at {datetime.now(timezone.utc).isoformat()}")
    
    async with httpx.AsyncClient() as client:  # shared client if needed
        for loc in LOCATIONS:
            try:
                traffic = await get_tomtom_flow(loc["lat"], loc["lon"])
                weather = await get_weather(loc["lat"], loc["lon"])
                
                row = {
                    "location_name": loc["name"],
                    "lat": loc["lat"],
                    "lon": loc["lon"],
                    "current_speed": traffic["current_speed"],
                    "free_flow_speed": traffic["free_flow_speed"],
                    "speed_ratio": traffic["speed_ratio"],
                    "delay_seconds": traffic["delay_seconds"],
                    "weather_main": weather["weather_main"],
                    "rain_mm": weather["rain_mm"],
                    "raw_data": {
                        "tomtom": traffic["raw"],
                        "weather": weather["raw"]
                    }
                }
                
                supabase.table("traffic_logs").insert(row).execute()
                print(f"✓ {loc['name']} | Speed ratio: {traffic['speed_ratio']} | Rain: {weather['rain_mm']}mm")
                
            except Exception as e:
                print(f"✗ Failed {loc['name']}: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(collect_and_store())