from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


def save_confusion_matrix(matrix: list[list[int]], class_names: tuple[str, ...], output: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(np.asarray(matrix), display_labels=class_names).plot(
        ax=axis, colorbar=False, xticks_rotation=45
    )
    figure.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def save_metric_plot(rows: list[dict[str, Any]], output: str | Path) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty or not {"model", "qubits", "macro_f1"}.issubset(frame.columns):
        raise ValueError("Metric rows must include model, qubits and macro_f1")
    figure, axis = plt.subplots(figsize=(7, 4))
    for model, subset in frame.groupby("model"):
        axis.plot(subset["qubits"], subset["macro_f1"], marker="o", label=model)
    axis.set(xlabel="Qubits", ylabel="Macro-F1", ylim=(0, 1), title="Validation/test macro-F1")
    axis.legend()
    figure.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)

