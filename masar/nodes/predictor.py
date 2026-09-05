"""Predictor node with a trained-model and no-model fallback."""

import os

import joblib
import numpy as np

from masar.state import MasarState
from masar.features import FEATURE_ORDER

MODEL_PATH = os.getenv("MASAR_MODEL_PATH", "models/congestion_xgb.pkl")
CONFIDENCE_THRESHOLD = 0.7

_model = None
_model_checked = False


def _load_model():
    """Load the model once, returning None until a model is available."""
    global _model, _model_checked
    if not _model_checked:
        _model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
        _model_checked = True
    return _model


def _build_feature_vector(state: MasarState) -> np.ndarray:
    return np.array([[state[name] for name in FEATURE_ORDER],])


def _heuristic_predict(state: MasarState) -> tuple[float, float]:
    ratio = state["congestion_ratio"]
    predicted = max(0.0, min(1.0, 1 - ratio))
    return predicted, 0.4


def predictor_node(state: MasarState) -> dict:
    model = _load_model()

    if model is None:
        predicted, confidence = _heuristic_predict(state)
    else:
        proba = model.predict_proba(_build_feature_vector(state))[0]
        predicted = float(proba[1])
        confidence = float(max(proba))

    return {
        "predicted_congestion": predicted,
        "prediction_confidence": confidence,
    }