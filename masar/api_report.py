"""Opt-in, redacted reports for external API and model calls."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

MAX_VALUE_LENGTH = 4000
SECRET_WORDS = ("key", "token", "secret", "password", "authorization", "appid")


def _safe(value: Any, key: str = "") -> Any:
    if any(word in key.lower() for word in SECRET_WORDS):
        return "[REDACTED]"
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif hasattr(value, "content"):
        value = value.content
    elif hasattr(value, "data"):
        value = {"data": value.data}
    if isinstance(value, dict):
        return {str(item_key): _safe(item_value, str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        result = value
    else:
        result = str(value)
    if isinstance(result, str) and len(result) > MAX_VALUE_LENGTH:
        return f"{result[:MAX_VALUE_LENGTH]}... [truncated]"
    return result


def record_call(
    service: str,
    operation: str,
    request: Any,
    response: Any = None,
    *,
    status: str = "success",
    duration_ms: float | None = None,
    error: Exception | str | None = None,
) -> None:
    """Append one redacted call record when MASAR_API_REPORT_PATH is set."""
    report_path = os.getenv("MASAR_API_REPORT_PATH")
    if not report_path:
        return
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": service,
        "operation": operation,
        "status": status,
        "duration_ms": round(duration_ms, 2) if duration_ms is not None else None,
        "request": _safe(request),
        "response": _safe(response),
        "error": str(error) if error else None,
    }
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as report:
        report.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")


def measured_call(
    service: str,
    operation: str,
    request: Any,
    function: Callable[[], Any],
) -> Any:
    started = time.perf_counter()
    try:
        response = function()
    except Exception as error:
        record_call(service, operation, request, status="error", duration_ms=(time.perf_counter() - started) * 1000, error=error)
        raise
    record_call(service, operation, request, response, duration_ms=(time.perf_counter() - started) * 1000)
    return response