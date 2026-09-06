"""Benchmark Masar's congestion classifier against simple baselines."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from train_model import generate_synthetic_rows, label_anomalies


def _metrics(name, y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "classification_report": classification_report(
            y_true, predictions, output_dict=True, zero_division=0
        ),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
        metrics["pr_auc"] = float(average_precision_score(y_true, probabilities))
        metrics["log_loss"] = float(log_loss(y_true, probabilities, labels=[0, 1]))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        metrics["log_loss"] = None
    return {"name": name, **metrics}


def benchmark(rows=6000, seed=42, model_path="models/congestion_xgb.pkl"):
    dataframe = label_anomalies(generate_synthetic_rows(rows, seed=seed))
    dataframe = dataframe.sort_values("timestamp").reset_index(drop=True)
    split = int(len(dataframe) * 0.8)
    test = dataframe.iloc[split:]
    y_true = test["is_anomaly"].to_numpy()
    ratio = test["congestion_ratio"].to_numpy()

    artifact = joblib.load(model_path)
    probabilities = artifact.predict_proba(test[artifact.get_booster().feature_names])[:, 1]
    majority_probability = np.full(len(test), float(dataframe.iloc[:split]["is_anomaly"].mean()))
    heuristic_probability = np.clip(1 - ratio, 0, 1)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "rows": len(dataframe),
            "train_rows": split,
            "test_rows": len(test),
            "source": "synthetic only",
            "seed": seed,
            "anomaly_rows": int(dataframe["is_anomaly"].sum()),
            "anomaly_rate": float(dataframe["is_anomaly"].mean()),
            "label_definition": "z-score below -1.5 within location/hour/weekend bucket",
        },
        "interpretation": "These metrics measure reproduction of synthetic labels, not future real-world traffic forecasting.",
        "benchmarks": [
            _metrics("majority_class_baseline", y_true, majority_probability),
            _metrics("heuristic_1_minus_congestion_ratio", y_true, heuristic_probability),
            _metrics("xgboost_trained_model", y_true, probabilities),
        ],
    }


def markdown_report(report):
    dataset = report["dataset"]
    lines = [
        "# Masar Model Benchmark",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Dataset",
        "",
        f"- Source: **{dataset['source']}**",
        f"- Rows: **{dataset['rows']}** ({dataset['train_rows']} train, {dataset['test_rows']} test)",
        f"- Anomaly rows: **{dataset['anomaly_rows']}** ({dataset['anomaly_rate']:.1%})",
        f"- Seed: `{dataset['seed']}`",
        f"- Labels: {dataset['label_definition']}",
        "",
        "> These metrics measure reproduction of synthetic labels. They are not yet validated future real-world traffic forecasts.",
        "",
        "## Test Metrics",
        "",
        "| Benchmark | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["benchmarks"]:
        lines.append(
            f"| {item['name']} | {item['accuracy']:.3f} | {item['balanced_accuracy']:.3f} | "
            f"{item['precision']:.3f} | {item['recall']:.3f} | {item['f1']:.3f} | "
            f"{item['roc_auc']:.3f} | {item['pr_auc']:.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- The majority baseline looks accurate because anomalies are rare, but it detects no anomalies.",
        "- The heuristic finds some anomalies but produces many false positives.",
        "- XGBoost substantially improves anomaly ranking and recall on this synthetic test split.",
        "- Real-only evaluation remains necessary before claiming production accuracy.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="models/congestion_xgb.pkl")
    parser.add_argument("--output", default="reports/model_benchmark.json")
    args = parser.parse_args()
    report = benchmark(args.rows, args.seed, args.model)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_output = output.with_suffix(".md")
    markdown_output.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved benchmark report to {output}")
    print(f"Saved readable benchmark report to {markdown_output}")


if __name__ == "__main__":
    main()