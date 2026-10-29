from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    if probabilities.ndim != 2 or len(y_true) != len(probabilities):
        raise ValueError("Expected probabilities with shape (n_samples, n_classes)")
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        if mask.any():
            error += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(error)


def evaluate_predictions(
    y_true: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    labels = np.arange(len(class_names))
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "per_class": classification_report(
            y_true, predictions, labels=labels, target_names=list(class_names),
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=labels).tolist(),
        "expected_calibration_error": expected_calibration_error(y_true, probabilities),
    }
    try:
        result["macro_auroc"] = float(
            roc_auc_score(y_true, probabilities, labels=labels, multi_class="ovr", average="macro")
        )
    except ValueError:
        result["macro_auroc"] = None
    return result


def timed_predict(model: Any, values: np.ndarray) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    output = model.predict_proba(values)
    elapsed = time.perf_counter() - started
    return np.asarray(output), elapsed / max(len(values), 1)
