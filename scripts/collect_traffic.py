"""Collect current traffic and weather signals for all monitored corridors."""

import asyncio
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from supabase import create_client

from masar.live_data import LOCATIONS, fetch_and_store

load_dotenv()


async def collect_and_store():
    print(f"Starting collection at {datetime.now(timezone.utc).isoformat()}")
    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    successful_rows = 0

    for location in LOCATIONS:
        try:
            row = await fetch_and_store(location["label"], supabase)
            successful_rows += 1
            print(f"OK {location['name']} | Speed ratio: {row['speed_ratio']} | Rain: {row['rain_mm']}mm")
        except Exception as error:
            print(f"FAILED {location['name']}: {error}")

    print(f"Stored {successful_rows}/{len(LOCATIONS)} rows in Supabase traffic_logs.")
    if successful_rows == 0:
        raise RuntimeError("No traffic data was stored in Supabase.")
    if successful_rows < len(LOCATIONS):
        raise RuntimeError(f"Only {successful_rows}/{len(LOCATIONS)} traffic rows were stored in Supabase.")


if __name__ == "__main__":
    asyncio.run(collect_and_store())
