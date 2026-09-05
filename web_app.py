"""Local Masar web app for testing the commute recommendation workflow."""

import json
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
FRONTEND = ROOT / "frontend"
GROQ_MODEL = os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b"


class DemoContextLLM:
    def invoke(self, prompt):
        return type(
            "Response",
            (),
            {"content": "Heavy traffic is likely related to the reported accident."},
        )()


class DemoSynthesisLLM:
    def invoke(self, prompt):
        from masar.nodes.synthesis import SynthesisOutput

        return SynthesisOutput(
            message_en="Use the recommended route and allow extra time for congestion.",
            message_ar="استخدم المسار الموصى به واترك وقتًا إضافيًا للازدحام.",
        )


def make_state(payload):
    return {
        "location": payload.get("location", "Sheikh Zayed Road"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "current_speed": float(payload.get("current_speed", 20)),
        "free_flow_speed": float(payload.get("free_flow_speed", 90)),
        "congestion_ratio": float(payload.get("congestion_ratio", 0.22)),
        "incidents": payload.get("incidents") or [
            "Accident reported near Trade Centre exit"
        ],
        "temperature": float(payload.get("temperature", 40)),
        "weather_condition": payload.get("weather_condition", "Clear"),
        "is_raining": bool(payload.get("is_raining", False)),
        "predicted_congestion": None,
        "prediction_confidence": None,
        "is_anomaly": False,
        "route_path": "fast",
        "origin": payload.get("origin", "Marina"),
        "destination": payload.get("destination", "Business Bay"),
        "context_notes": None,
        "nearby_events": [],
        "social_signal": None,
        "recommended_mode": None,
        "recommended_route": [],
        "message_en": None,
        "message_ar": None,
    }


def run_recommendation(payload):
    from masar.graph import build_graph

    demo = payload.get("demo", True)
    if demo:
        graph = build_graph(DemoContextLLM(), DemoSynthesisLLM())
    else:
        graph = build_graph()
    return graph.invoke(make_state(payload))


def traffic_snapshot():
    from supabase import create_client

    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    response = (
        client.table("traffic_logs")
        .select("location_name,current_speed,free_flow_speed,speed_ratio,weather_main,created_at")
        .order("created_at", desc=True)
        .limit(8)
        .execute()
    )
    return response.data or []


class MasarHandler(BaseHTTPRequestHandler):
    def _send(self, status, body, content_type="application/json"):
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(
                200,
                (FRONTEND / "index.html").read_text(encoding="utf-8"),
                "text/html",
            )
            return
        if path == "/api/traffic":
            try:
                self._send(200, json.dumps({"rows": traffic_snapshot()}, default=str))
            except Exception as error:
                self._send(503, json.dumps({"error": str(error)}))
            return
        self._send(404, json.dumps({"error": "Not found"}))

    def do_POST(self):
        if urlparse(self.path).path != "/api/recommend":
            self._send(404, json.dumps({"error": "Not found"}))
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            try:
                result = run_recommendation(payload)
            except Exception as error:
                if payload.get("demo", True):
                    raise
                result = run_recommendation({**payload, "demo": True})
                result["live_error"] = str(error)
                result["used_fallback"] = True
            self._send(200, json.dumps({"result": result}, default=str))
        except Exception as error:
            self._send(500, json.dumps({"error": str(error)}))

    def log_message(self, format, *args):
        print(f"[masar] {self.address_string()} - {format % args}")


if __name__ == "__main__":
    port = int(os.getenv("MASAR_PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), MasarHandler)
    print(f"Masar is running at http://127.0.0.1:{port}")
    print("Demo mode is available in the browser; Live mode uses Groq.")
    server.serve_forever()