import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import web_app
from web_app import MasarHandler


def test_frontend_contains_current_map_ui():
    html = (Path(__file__).parents[1] / "frontend" / "index.html").read_text(encoding="utf-8")
    assert 'id="route-map"' in html
    assert "Route intelligence" in html


def test_health_endpoint():
    server = ThreadingHTTPServer(("127.0.0.1", 0), MasarHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_port}/health"
        ) as response:
            assert response.status == 200
            assert json.loads(response.read()) == {"status": "ok"}
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_locations_endpoint_exposes_the_full_dubai_catalog():
    server = ThreadingHTTPServer(("127.0.0.1", 0), MasarHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_port}/api/locations"
        ) as response:
            locations = json.loads(response.read())
        labels = {location["label"] for location in locations}
        assert len(locations) >= 40
        assert {"Marina", "Deira", "Mirdif", "Dubai Silicon Oasis"}.issubset(labels)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_recommendation_build_refreshes_live_conditions(monkeypatch):
    calls = []

    def refresh_live_fields(location, supabase_client=None):
        calls.append((location, supabase_client))
        return (
            {
                "current_speed": 18.0,
                "free_flow_speed": 90.0,
                "congestion_ratio": 0.2,
                "weather_condition": "Rain",
                "is_raining": True,
                "temperature": 29.0,
            },
            {"created_at": "2026-09-08T10:00:00+00:00"},
        )

    monkeypatch.setattr(web_app.live_traffic, "refresh_live_fields", refresh_live_fields)
    result = web_app.run_recommendation(
        {"demo": True, "use_live_traffic": True, "origin": "Marina", "destination": "Business Bay"}
    )

    assert len(calls) == 1
    assert calls[0][0] == "Marina"
    assert result["used_live_traffic"] is True
    assert result["current_speed"] == 18.0
    assert result["temperature"] == 29.0