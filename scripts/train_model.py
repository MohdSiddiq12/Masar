"""Train the congestion classifier from Supabase rows and synthetic history."""

import argparse
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from supabase import create_client

from masar.features import features_to_array

DEFAULT_MODEL_PATH = "models/congestion_xgb.pkl"
CONGESTION_RATIO_THRESHOLD = 0.7


def fetch_traffic_rows() -> list[dict]:
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY are required for real data.")
    return create_client(url, key).table("traffic_logs").select("*").execute().data or []


def make_synthetic_rows(count: int, seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(count):
        free_flow = float(rng.choice([80, 100, 110]))
        ratio = float(rng.uniform(0.15, 0.98))
        rows.append(
            {
                "current_speed": free_flow * ratio,
                "free_flow_speed": free_flow,
                "speed_ratio": ratio,
                "temperature": float(rng.uniform(22, 45)),
                "rain_mm": float(rng.choice([0, 0, 0, 2, 8])),
                "weather_main": "Rain" if index % 5 == 0 else "Clear",
            }
        )
    return rows


def train(rows: list[dict], output_path: str = DEFAULT_MODEL_PATH):
    if not rows:
        raise ValueError("No training rows were supplied.")
    X = features_to_array(rows)
    ratios = np.asarray([
        float(row.get("congestion_ratio", row.get("speed_ratio"))) for row in rows
    ])
    y = (ratios < CONGESTION_RATIO_THRESHOLD).astype(int)
    if len(set(y)) < 2:
        raise ValueError("Training data must contain both congested and uncongested rows.")

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(X, y)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(model, destination)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-rows", type=int, default=200)
    parser.add_argument("--include-real", action="store_true")
    parser.add_argument("--real-only", action="store_true")
    parser.add_argument("--output", default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    if args.real_only:
        rows = fetch_traffic_rows()
    else:
        rows = fetch_traffic_rows() if args.include_real else []
        rows += make_synthetic_rows(args.synthetic_rows)
    model = train(rows, args.output)
    print(f"Trained model on {len(rows)} rows; classes={list(model.classes_)}; saved to {args.output}")


if __name__ == "__main__":
    main()