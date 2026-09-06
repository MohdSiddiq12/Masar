"""Run isolated, offline health checks for the Masar components.

The default run never calls Groq, Supabase, TomTom, OpenWeather, or Reddit.
Use ``--include-network`` only when those external checks are explicitly
desired and the required environment variables are configured.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str


class FakeContextLLM:
    def invoke(self, prompt):
        return type("Response", (), {"content": "Offline context check completed."})()


class FakeSynthesisLLM:
    def invoke(self, prompt):
        from masar.nodes.synthesis import SynthesisOutput

        return SynthesisOutput(
            message_en="Use the recommended route.",
            message_ar="استخدم المسار الموصى به.",
        )


def sample_state(**overrides):
    state = {
        "location": "Marina",
        "timestamp": "2026-09-06T18:00:00Z",
        "current_speed": 20.0,
        "free_flow_speed": 90.0,
        "congestion_ratio": 20 / 90,
        "incidents": [],
        "temperature": 40.0,
        "weather_condition": "Clear",
        "is_raining": False,
        "predicted_congestion": None,
        "prediction_confidence": None,
        "is_anomaly": False,
        "route_path": "fast",
        "origin": "Marina",
        "destination": "Business Bay",
        "context_notes": None,
        "nearby_events": [],
        "social_signal": None,
        "recommended_mode": None,
        "recommended_route": [],
        "message_en": None,
        "message_ar": None,
    }
    state.update(overrides)
    return state


def check_synthetic_data():
    from train_model import generate_synthetic_rows

    dataframe = generate_synthetic_rows(100, seed=7)
    required = {"timestamp", "location", "current_speed", "congestion_ratio", "source"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise AssertionError(f"missing columns: {sorted(missing)}")
    if set(dataframe["source"]) != {"synthetic"}:
        raise AssertionError("synthetic rows are not marked synthetic")
    return (
        f"input rows=100, seed=7 -> output rows={len(dataframe)}, "
        f"columns={len(dataframe.columns)}, source=synthetic"
    )


def check_features():
    from masar.features import FEATURE_ORDER, row_to_features

    values = row_to_features({
        "current_speed": 45,
        "free_flow_speed": 90,
        "speed_ratio": 0.5,
        "raw_data": {"weather": {"main": {"temp": 35}}},
        "weather_main": "Rain",
        "rain_mm": 1,
    })
    if len(values) != len(FEATURE_ORDER) or not np.isfinite(values).all():
        raise AssertionError(f"invalid feature vector: {values}")
    return (
        "input speed=45, free_flow=90, ratio=0.5, rain=1 -> "
        f"output={values}; order={', '.join(FEATURE_ORDER)}"
    )


def check_predictor():
    import masar.nodes.predictor as predictor_module

    predictor_module._model = None
    predictor_module._model_checked = False
    model_path = predictor_module.MODEL_PATH
    result = predictor_module.predictor_node(sample_state())
    if predictor_module._model is None:
        source = "heuristic fallback"
        if result["prediction_confidence"] >= predictor_module.CONFIDENCE_THRESHOLD:
            raise AssertionError("heuristic confidence should be below router threshold")
    else:
        source = f"trained model ({model_path})"
    if not 0 <= result["predicted_congestion"] <= 1:
        raise AssertionError(f"invalid prediction: {result}")
    if not 0 <= result["prediction_confidence"] <= 1:
        raise AssertionError(f"invalid confidence: {result}")
    return (
        f"input location=Marina, speed=20/90, clear -> used {source}; "
        f"output prediction={result['predicted_congestion']:.3f}, "
        f"confidence={result['prediction_confidence']:.3f}"
    )


def check_router():
    from masar.nodes.router import router_node

    low = router_node({"prediction_confidence": 0.3})
    high = router_node({"prediction_confidence": 0.9})
    if low.goto != "context_agent" or high.goto != "route_optimizer":
        raise AssertionError(f"unexpected routes: low={low.goto}, high={high.goto}")
    return (
        f"input confidence=0.30 -> {low.goto}; "
        f"input confidence=0.90 -> {high.goto}"
    )


def check_context():
    from masar.nodes.context import context_node

    result = context_node(sample_state(), llm=FakeContextLLM())
    if result["context_notes"] != "Offline context check completed.":
        raise AssertionError(f"unexpected context result: {result}")
    return (
        "input location=Marina, ratio=0.22, clear, no incidents -> "
        f"output notes={result['context_notes']!r}; network calls=0"
    )


def check_optimizer():
    from masar.nodes.optimizer import route_optimizer_node

    result = route_optimizer_node(sample_state(predicted_congestion=0.8))
    if not result["recommended_route"] or result["recommended_mode"] != "metro":
        raise AssertionError(f"unexpected optimizer result: {result}")
    return (
        "input origin=Marina, destination=Business Bay, congestion=0.80 -> "
        f"output route={' -> '.join(result['recommended_route'])}, "
        f"mode={result['recommended_mode']}"
    )


def check_synthesis():
    from masar.nodes.synthesis import synthesis_node

    result = synthesis_node(
        sample_state(recommended_route=["Marina", "Business Bay"], recommended_mode="drive"),
        llm=FakeSynthesisLLM(),
    )
    if not result["message_en"] or not result["message_ar"]:
        raise AssertionError(f"missing bilingual output: {result}")
    return (
        "input route=Marina -> Business Bay, mode=drive -> "
        f"output English={result['message_en']!r}, Arabic={result['message_ar']!r}"
    )


def check_graph():
    from masar.graph import build_graph

    result = build_graph(FakeContextLLM(), FakeSynthesisLLM()).invoke(sample_state())
    required = {"predicted_congestion", "prediction_confidence", "recommended_route", "message_en", "message_ar"}
    missing = required.difference(result)
    if missing:
        raise AssertionError(f"missing graph outputs: {sorted(missing)}")
    return (
        "input location=Marina, speed=20/90, clear -> "
        f"output prediction={result['predicted_congestion']:.3f}, "
        f"confidence={result['prediction_confidence']:.3f}, "
        f"path={result['route_path']}, mode={result['recommended_mode']}, "
        f"route={' -> '.join(result['recommended_route'])}"
    )


def check_network():
    from train_model import fetch_real_rows

    dataframe = fetch_real_rows()
    return f"input select traffic_logs -> output {len(dataframe)} real rows; writes=0"


CHECKS: list[tuple[str, Callable[[], str]]] = [
    ("Synthetic data generator", check_synthetic_data),
    ("Feature engineering", check_features),
    ("Predictor", check_predictor),
    ("Confidence router", check_router),
    ("Context component", check_context),
    ("Route optimizer", check_optimizer),
    ("Bilingual synthesis", check_synthesis),
    ("Full LangGraph pipeline", check_graph),
]


def run_checks(include_network: bool = False) -> list[CheckResult]:
    from masar.api_report import record_call

    checks = list(CHECKS)
    if include_network:
        checks.append(("Supabase read-only connection", check_network))
    results = []
    print("Masar component diagnostics")
    print(f"Mode: {'offline + Supabase read-only' if include_network else 'offline (no network)'}")
    print("=" * 72)
    for name, check in checks:
        try:
            details = check()
            result = CheckResult(name, True, details)
        except Exception as error:
            result = CheckResult(name, False, f"{type(error).__name__}: {error}")
        results.append(result)
        record_call(
            "component",
            name,
            {"mode": "network" if include_network else "offline"},
            result.details,
            status="success" if result.passed else "error",
        )
        print(f"{'PASS' if result.passed else 'FAIL':4} | {name}: {result.details}")
    passed = sum(result.passed for result in results)
    print("=" * 72)
    print(f"Summary: {passed}/{len(results)} checks passed")
    if not all(result.passed for result in results):
        print("Action: inspect each FAIL above; the process exits with status 1.")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-network",
        action="store_true",
        help="also perform the read-only Supabase check; never writes data",
    )
    parser.add_argument(
        "--report-path",
        default=os.getenv("MASAR_API_REPORT_PATH"),
        help="write redacted API/model call records to this JSONL file",
    )
    args = parser.parse_args()
    if args.report_path:
        os.environ["MASAR_API_REPORT_PATH"] = args.report_path
    return 0 if all(result.passed for result in run_checks(args.include_network)) else 1


if __name__ == "__main__":
    raise SystemExit(main())