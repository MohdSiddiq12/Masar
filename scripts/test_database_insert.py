import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

supabase = create_client(url, key)

# Test insert
data = {
    "location_name": "test_location",
    "lat": 25.2048,
    "lon": 55.2708,
    "current_speed": 45,
    "free_flow_speed": 100,
    "speed_ratio": 0.45,
    "delay_seconds": 120,
    "weather_main": "Clear",
    "rain_mm": 0
}

result = supabase.table("traffic_logs").insert(data).execute()
print("Insert successful!")
print(result)