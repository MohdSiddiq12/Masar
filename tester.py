import json
import os
import shutil
import subprocess
from collections import Counter

from dotenv import load_dotenv

load_dotenv()

# Current model available to the configured Groq account.
GROQ_MODEL = os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b"
os.environ["GROQ_MODEL"] = GROQ_MODEL


def run_check(name, function):
    print(f"\n{'=' * 20} {name} {'=' * 20}")
    try:
        result = function()
        print("PASS")
        return result
    except Exception as error:
        print(f"FAIL: {type(error).__name__}: {error}")
        return None


def check_environment():
    required = [
        "GROQ_API_KEY",
        "SUPABASE_URL",
        "SUPABASE_KEY",
        "TOMTOM_API_KEY",
        "OPENWEATHER_API_KEY",
    ]

    print("Groq model:", GROQ_MODEL)

    for variable in required:
        print(f"{variable}: {'configured' if os.getenv(variable) else 'missing'}")


def check_groq():
    from langchain_groq import ChatGroq

    llm = ChatGroq(model=GROQ_MODEL, temperature=0.2)
    response = llm.invoke("Say hello in one short sentence.")
    print(response.content)


def check_supabase():
    from supabase import create_client

    client = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_KEY"],
    )

    response = (
        client.table("traffic_logs")
        .select("*")
        .limit(3)
        .execute()
    )

    rows = response.data or []
    print("Rows returned:", len(rows))

    if rows:
        print("Columns:", sorted(rows[0].keys()))
        print(json.dumps(rows[0], indent=2, default=str))


def check_training_data():
    from scripts.train_model import fetch_real_rows
    from train_model import label_anomalies

    dataframe = fetch_real_rows()
    labeled = label_anomalies(dataframe)

    print("Shape:", dataframe.shape)
    print("Columns:", list(dataframe.columns))
    print("Null values:", int(dataframe.isna().sum().sum()))
    print("Labels:", labeled["is_anomaly"].value_counts().to_dict())

    required_columns = {
        "location",
        "current_speed",
        "free_flow_speed",
        "congestion_ratio",
        "temperature",
        "is_raining",
    }

    missing = required_columns.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Missing training columns: {sorted(missing)}")

    if labeled["is_anomaly"].nunique() < 2:
        print(
            "WARNING: real-only training cannot proceed because the current "
            "dataset contains only one class."
        )


def build_state():
    return {
        "location": "Sheikh Zayed Road",
        "timestamp": "2026-09-05T18:00:00Z",
        "current_speed": 20.0,
        "free_flow_speed": 90.0,
        "congestion_ratio": 0.22,
        "incidents": [
            "Accident reported near Trade Centre exit",
        ],
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


def check_predictor():
    from masar.nodes.predictor import predictor_node

    result = predictor_node(build_state())
    print(result)

    assert 0 <= result["predicted_congestion"] <= 1
    assert 0 <= result["prediction_confidence"] <= 1


def check_router():
    from masar.nodes.router import router_node

    low_confidence = router_node({"prediction_confidence": 0.3})
    high_confidence = router_node({"prediction_confidence": 0.9})
    missing_confidence = router_node({})

    print("Low confidence:", low_confidence)
    print("High confidence:", high_confidence)
    print("Missing confidence:", missing_confidence)


def check_context():
    from masar.nodes.context import context_node

    result = context_node(build_state())
    print("Context:", result["context_notes"])
    print("Events:", result["nearby_events"])
    print("Social signal:", result["social_signal"])


def check_optimizer():
    from masar.nodes.optimizer import route_optimizer_node

    result = route_optimizer_node(
        {
            "predicted_congestion": 0.7,
            "origin": "Marina",
            "destination": "Business Bay",
        }
    )

    print(result)


def check_synthesis():
    from masar.nodes.synthesis import synthesis_node

    state = {
        "recommended_route": [
            "Marina",
            "Al Khail",
            "Business Bay",
        ],
        "recommended_mode": "drive_to_metro",
        "context_notes": (
            "Heavy congestion due to an accident near Trade Centre."
        ),
    }

    result = synthesis_node(state)

    print("English:", result["message_en"])
    print("Arabic:", result["message_ar"])


def check_full_graph():
    from masar.graph import build_graph

    result = build_graph().invoke(build_state())

    print("Route:", result["recommended_route"])
    print("Mode:", result["recommended_mode"])
    print("Context:", result["context_notes"])
    print("English:", result["message_en"])
    print("Arabic:", result["message_ar"])


def check_github_cli():
    if shutil.which("gh") is None:
        raise RuntimeError("GitHub CLI is not installed or not on PATH.")

    result = subprocess.run(
        ["gh", "auth", "status"],
        capture_output=True,
        text=True,
        check=False,
    )

    print(result.stdout or result.stderr)

    if result.returncode != 0:
        raise RuntimeError("GitHub CLI is not authenticated.")


if __name__ == "__main__":
    run_check("Environment", check_environment)
    run_check("Groq connectivity", check_groq)
    run_check("Supabase traffic_logs", check_supabase)
    run_check("Training data normalization", check_training_data)
    run_check("Predictor", check_predictor)
    run_check("Router", check_router)
    run_check("Context agent", check_context)
    run_check("Route optimizer", check_optimizer)
    run_check("Bilingual synthesis", check_synthesis)
    run_check("Full graph", check_full_graph)
    run_check("GitHub CLI access", check_github_cli)