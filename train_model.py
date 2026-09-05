"""
Trains an XGBoost classifier to predict whether current conditions are
ANOMALOUS for that location and time -- not just "is it rush hour",
which would mislabel every weekday evening as broken.

Synthetic data is the default so this pipeline is fully testable
without network access or real data volume. --include-real pulls from
Supabase and appends it; --real-only uses only Supabase data. The real
path requires SUPABASE_URL / SUPABASE_KEY and hasn't been exercised in
this environment -- no network path to Supabase here. Test that flag
in your own environment before trusting it.

LABEL DEFINITION: for each (location, hour, is_weekend) bucket, compute
the mean and std of congestion_ratio, then flag a row anomalous if its
ratio is more than ANOMALY_Z_THRESHOLD standard deviations below that
bucket's mean. This needs enough repeated observations per bucket to
be meaningful -- with only a few days of real data, most buckets won't
have enough history yet. Synthetic mode exists to prove the mechanism
correctly while real coverage grows.
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from masar.features import FEATURE_ORDER

LOCATIONS = ["Sheikh Zayed Road", "Al Khail", "Business Bay", "Marina", "Airport"]
ANOMALY_Z_THRESHOLD = 1.5


def generate_synthetic_rows(n_rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2026-08-01", tz="UTC")

    for i in range(n_rows):
        timestamp = start + pd.Timedelta(minutes=10 * i)
        location = LOCATIONS[i % len(LOCATIONS)]
        hour = timestamp.hour
        is_weekend = timestamp.dayofweek in (4, 5)  # UAE weekend: Fri/Sat

        free_flow_speed = 90.0 + rng.normal(0, 3)
        is_rush_hour = (not is_weekend) and (7 <= hour <= 9 or 17 <= hour <= 20)
        base_ratio = 0.45 if is_rush_hour else 0.85

        is_raining = rng.random() < 0.04  # rare, Dubai-specific
        rain_penalty = 0.35 if is_raining else 0.0
        incident_penalty = 0.3 if rng.random() < 0.03 else 0.0  # hard cases

        noise = rng.normal(0, 0.05)
        ratio = float(np.clip(base_ratio - rain_penalty - incident_penalty + noise, 0.05, 1.0))
        current_speed = free_flow_speed * ratio
        temperature = 32 + 8 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 1.5)

        rows.append({
            "timestamp": timestamp,
            "location": location,
            "current_speed": current_speed,
            "free_flow_speed": free_flow_speed,
            "congestion_ratio": ratio,
            "temperature": temperature,
            "is_raining": bool(is_raining),
            "hour": hour,
            "is_weekend": is_weekend,
        })

    return pd.DataFrame(rows)


def fetch_real_rows() -> pd.DataFrame:
    """Pulls traffic_logs from Supabase, normalized to this trainer's
    column names. NOT exercised in this sandbox. Confirm the rename
    map against a real row before trusting it -- e.g. your table may
    store `speed_ratio` where this expects `congestion_ratio`.
    """
    from supabase import create_client

    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    response = client.table("traffic_logs").select("*").order("created_at").execute()
    df = pd.DataFrame(response.data)

    df = df.rename(columns={
        "created_at": "timestamp",
        "speed_ratio": "congestion_ratio",  # confirm this matches your real schema
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["hour"] = df["timestamp"].dt.hour
    df["is_weekend"] = df["timestamp"].dt.dayofweek.isin([4, 5])
    return df


def label_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    baseline = (
        df.groupby(["location", "hour", "is_weekend"])["congestion_ratio"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "baseline_mean", "std": "baseline_std"})
    )
    df = df.join(baseline, on=["location", "hour", "is_weekend"])
    fallback_std = df["congestion_ratio"].std() or 1.0
    df["baseline_std"] = df["baseline_std"].fillna(fallback_std).replace(0, fallback_std)

    z = (df["congestion_ratio"] - df["baseline_mean"]) / df["baseline_std"]
    df["is_anomaly"] = (z < -ANOMALY_Z_THRESHOLD).astype(int)
    return df


def train(df: pd.DataFrame) -> XGBClassifier:
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)  # time-based split -- not random, avoids leakage
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train = train_df[FEATURE_ORDER].astype(float)
    y_train = train_df["is_anomaly"]
    X_test = test_df[FEATURE_ORDER].astype(float)
    y_test = test_df["is_anomaly"]

    n_pos = max(int(y_train.sum()), 1)
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos  # counteracts class imbalance

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    if len(X_test) and y_test.nunique() > 1:
        preds = model.predict(X_test)
        print(classification_report(y_test, preds, zero_division=0))
    else:
        print("Test split too small or single-class -- skipping classification report.")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-rows", type=int, default=2000)
    parser.add_argument("--include-real", action="store_true")
    parser.add_argument("--real-only", action="store_true")
    parser.add_argument("--output", default="models/congestion_xgb.pkl")
    args = parser.parse_args()

    frames = []
    if not args.real_only:
        frames.append(generate_synthetic_rows(args.synthetic_rows))
    if args.include_real or args.real_only:
        frames.append(fetch_real_rows())

    df = pd.concat(frames, ignore_index=True)
    df = label_anomalies(df)

    if df["is_anomaly"].nunique() < 2:
        raise ValueError(
            "Training data must contain both normal and anomalous rows; "
            "collect more history or increase --synthetic-rows."
        )

    print(f"Training on {len(df)} rows, {int(df['is_anomaly'].sum())} labeled anomalous "
          f"({df['is_anomaly'].mean():.1%})")

    model = train(df)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
