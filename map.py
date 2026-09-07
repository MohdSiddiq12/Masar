import os
import json
import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TOMTOM_API_KEY")
BASE = "https://api.tomtom.com/routing/1/calculateRoute"

# Dubai sample points
ORIGIN = (25.0800, 55.1400)       # Dubai Marina
DESTINATION = (25.1850, 55.2650)  # Business Bay


def build_url(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    return f"{BASE}/{lat1},{lon1}:{lat2},{lon2}/json"


def fetch_routes(traffic=True, alternatives=True, travel_mode="car"):
    if not API_KEY:
        raise RuntimeError("TOMTOM_API_KEY missing in .env")

    params = {
        "key": API_KEY,
        "traffic": str(traffic).lower(),
        "travelMode": travel_mode,
        "routeType": "fastest",
        "computeBestOrder": "false",
        "maxAlternatives": 2 if alternatives else 0,
        "sectionType": "traffic",
        "instructionsType": "text",
    }

    url = build_url(ORIGIN, DESTINATION)
    with httpx.Client(timeout=30) as client:
        r = client.get(url, params=params)
        print("Status:", r.status_code)
        r.raise_for_status()
        return r.json()


def summarize(data):
    routes = data.get("routes", [])
    print(f"\nFound {len(routes)} route(s)\n")

    for i, route in enumerate(routes):
        summary = route.get("summary", {})
        legs = route.get("legs", [])
        points = []
        if legs:
            points = legs[0].get("points", [])

        length_m = summary.get("lengthInMeters")
        travel_s = summary.get("travelTimeInSeconds")
        traffic_delay_s = summary.get("trafficDelayInSeconds")
        departure = summary.get("departureTime")
        arrival = summary.get("arrivalTime")

        print(f"=== Route #{i+1} ===")
        print(f"Distance:        {length_m/1000:.2f} km" if length_m else "Distance: n/a")
        print(f"Travel time:     {travel_s/60:.1f} min" if travel_s else "Travel time: n/a")
        print(f"Traffic delay:   {traffic_delay_s/60:.1f} min" if traffic_delay_s is not None else "Traffic delay: n/a")
        print(f"Departure:       {departure}")
        print(f"Arrival:         {arrival}")
        print(f"Geometry points: {len(points)}")

        # print a few coordinates for map testing
        if points:
            print("Start point:", points[0])
            print("Mid point:  ", points[len(points)//2])
            print("End point:  ", points[-1])
        print()


def compare_with_and_without_traffic():
    print("\n---- WITH traffic=true ----")
    with_traffic = fetch_routes(traffic=True, alternatives=True)
    summarize(with_traffic)

    print("\n---- WITH traffic=false ----")
    no_traffic = fetch_routes(traffic=False, alternatives=False)
    summarize(no_traffic)

    # side-by-side first route
    s1 = with_traffic["routes"][0]["summary"]
    s0 = no_traffic["routes"][0]["summary"]
    print("==== Comparison (first route) ====")
    print(f"Travel time traffic ON : {s1.get('travelTimeInSeconds', 0)/60:.1f} min")
    print(f"Travel time traffic OFF: {s0.get('travelTimeInSeconds', 0)/60:.1f} min")
    print(f"Delay reported         : {s1.get('trafficDelayInSeconds', 0)/60:.1f} min")


def save_raw(data, path="tomtom_route_sample.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved raw response → {path}")


if __name__ == "__main__":
    # 1) Main exploration
    data = fetch_routes(traffic=True, alternatives=True)
    summarize(data)
    save_raw(data)

    # 2) Optional comparison
    compare_with_and_without_traffic()