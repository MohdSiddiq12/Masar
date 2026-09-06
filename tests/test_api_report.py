import json
from collections import defaultdict

from masar.api_report import measured_call, record_call


def _read_report(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _display_report(entries, expected_error=False):
    print("\nAPI CALL REPORT")
    print("=" * 88)
    grouped = defaultdict(list)
    for entry in entries:
        grouped[entry["service"]].append(entry)

    for service, calls in grouped.items():
        print(f"\nSERVICE: {service} ({len(calls)} call(s))")
        print("-" * 88)
        for number, call in enumerate(calls, start=1):
            print(f"[{number}] {call['timestamp']} | {call['operation']}")
            status = call["status"]
            if expected_error and status == "error":
                status += " (expected)"
            print(f"    status:   {status}")
            print(f"    duration: {call['duration_ms']} ms")
            print(f"    request:  {json.dumps(call['request'], ensure_ascii=False, default=str)}")
            print(f"    response: {json.dumps(call['response'], ensure_ascii=False, default=str)}")
            if call["error"]:
                print(f"    error:    {call['error']}")
    print("=" * 88)


def test_api_report_displays_each_service_call(tmp_path, monkeypatch):
    report = tmp_path / "calls.jsonl"
    monkeypatch.setenv("MASAR_API_REPORT_PATH", str(report))

    services = [
        ("groq", "context.invoke", {"model": "openai/gpt-oss-120b", "prompt": "Explain traffic", "api_key": "hidden"}, type("Response", (), {"content": "Traffic is heavy near Marina."})()),
        ("supabase", "traffic_logs.latest", {"table": "traffic_logs", "location": "Dubai_Marina", "select": "current_speed,speed_ratio"}, type("Response", (), {"data": [{"current_speed": 25, "speed_ratio": 0.28}]})()),
        ("tomtom", "traffic_flow", {"url": "https://api.tomtom.com/traffic", "params": {"point": "25.08,55.14", "key": "hidden"}}, {"currentSpeed": 25, "freeFlowSpeed": 90}),
        ("openweathermap", "current_weather", {"url": "https://api.openweathermap.org/data/2.5/weather", "params": {"lat": 25.08, "lon": 55.14, "appid": "hidden"}}, {"weather": [{"main": "Clear"}], "main": {"temp": 36}}),
        ("reddit", "new_posts", {"url": "https://www.reddit.com/r/dubai/new.json", "params": {"limit": 30}}, {"post_count": 4}),
    ]

    for service, operation, request, response in services:
        record_call(service, operation, request, response, duration_ms=12.3)

    entries = _read_report(report)
    _display_report(entries)

    assert {entry["service"] for entry in entries} == {"groq", "supabase", "tomtom", "openweathermap", "reddit"}
    assert len(entries) == 5
    assert entries[0]["request"]["api_key"] == "[REDACTED]"
    assert entries[2]["request"]["params"]["key"] == "[REDACTED]"
    assert entries[3]["request"]["params"]["appid"] == "[REDACTED]"
    assert entries[0]["response"] == "Traffic is heavy near Marina."
    assert entries[1]["response"] == {"data": [{"current_speed": 25, "speed_ratio": 0.28}]}


def test_api_report_displays_failed_call(tmp_path, monkeypatch):
    report = tmp_path / "failed-calls.jsonl"
    monkeypatch.setenv("MASAR_API_REPORT_PATH", str(report))

    def failing_call():
        raise TimeoutError("TomTom request timed out")

    try:
        measured_call("tomtom", "traffic_flow", {"url": "https://api.tomtom.com/traffic", "params": {"key": "hidden"}}, failing_call)
    except TimeoutError:
        pass
    else:
        raise AssertionError("the failing test call should raise")

    entries = _read_report(report)
    _display_report(entries, expected_error=True)
    assert len(entries) == 1
    assert entries[0]["status"] == "error"
    assert entries[0]["error"] == "TomTom request timed out"
    assert entries[0]["request"]["params"]["key"] == "[REDACTED]"
