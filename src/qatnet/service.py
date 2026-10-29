from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from sdn.telemetry import FlowTelemetry


@dataclass(frozen=True)
class DetectionResult:
    flow_id: str
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    attack_probability: float
    model_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "flow_id": self.flow_id,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "attack_probability": self.attack_probability,
            "model_version": self.model_version,
        }


class QATNetDetectionService:
    """In-process inference boundary for complete CIC flow-feature records."""

    def __init__(
        self,
        model: Any,
        class_names: Sequence[str],
        malicious_labels: set[str],
        model_version: str,
    ) -> None:
        if not class_names or len(set(class_names)) != len(class_names):
            raise ValueError("class_names must be nonempty and unique")
        if not malicious_labels.issubset(class_names):
            raise ValueError("malicious_labels must be included in class_names")
        expected = tuple(getattr(model.preprocessor, "input_feature_names", ()))
        if not expected:
            raise ValueError("Model has no fitted input-feature manifest")
        fitted_classes = tuple(getattr(model, "class_names", ()))
        if fitted_classes and fitted_classes != tuple(class_names):
            raise ValueError("Service class names do not match the fitted model")
        self.model = model
        self.class_names = tuple(class_names)
        self.malicious_labels = frozenset(malicious_labels)
        self.model_version = model_version
        self.expected_features = expected

    @classmethod
    def from_model_dir(
        cls, model_dir: str | Path, malicious_labels: set[str]
    ) -> "QATNetDetectionService":
        from qatnet.training import load_model

        model_dir = Path(model_dir)
        metadata_path = model_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        digest = hashlib.sha256()
        for path in sorted(model_dir.iterdir()):
            if path.is_file():
                digest.update(path.name.encode())
                digest.update(path.read_bytes())
        model_version = digest.hexdigest()[:16]
        model = load_model(model_dir)
        return cls(model, metadata["class_names"], malicious_labels, model_version)

    def predict(
        self,
        flows: Sequence[FlowTelemetry],
        flow_features: pd.DataFrame,
    ) -> list[DetectionResult]:
        if len(flows) != len(flow_features):
            raise ValueError("Each flow must have exactly one aligned feature record")
        missing = set(self.expected_features) - set(flow_features.columns)
        if missing:
            raise ValueError(
                f"Flow feature payload is incomplete; missing model features: {sorted(missing)}"
            )
        if not flows:
            return []
        features = flow_features.loc[:, self.expected_features]
        probabilities = np.asarray(
            self.model.predict_proba(features, use_cache=False), dtype=np.float64
        )
        if probabilities.shape != (len(flows), len(self.class_names)):
            raise ValueError("Model probability output does not match flow/class dimensions")
        if not np.isfinite(probabilities).all() or (probabilities < 0).any():
            raise ValueError("Model returned invalid probabilities")
        results = []
        for flow, row in zip(flows, probabilities):
            if not np.isclose(row.sum(), 1.0, atol=1e-5):
                raise ValueError("Model probability row does not sum to one")
            index = int(np.argmax(row))
            label = self.class_names[index]
            results.append(DetectionResult(
                flow_id=flow.flow_id,
                predicted_class=label,
                confidence=float(row[index]),
                probabilities={
                    name: float(row[class_index])
                    for class_index, name in enumerate(self.class_names)
                },
                attack_probability=float(sum(
                    row[class_index]
                    for class_index, name in enumerate(self.class_names)
                    if name in self.malicious_labels
                )),
                model_version=self.model_version,
            ))
        return results


def telemetry_from_cic_records(
    records: pd.DataFrame,
    id_prefix: str = "replay",
) -> list[FlowTelemetry]:
    required = {"Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Flow Duration", "Protocol"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"CIC replay records are missing telemetry fields: {sorted(missing)}")

    def finite_nonnegative(row: pd.Series, columns: tuple[str, ...], flow_index: int) -> float:
        try:
            values = [float(row[column]) for column in columns]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"CIC replay row {flow_index} has nonnumeric telemetry values in {columns}"
            ) from exc
        if not np.isfinite(values).all() or any(value < 0 for value in values):
            raise ValueError(
                f"CIC replay row {flow_index} has invalid telemetry values in {columns}"
            )
        return sum(values)

    flows = []
    for index, (_, row) in enumerate(records.iterrows()):
        protocol_value = pd.to_numeric(pd.Series([row["Protocol"]]), errors="coerce").iloc[0]
        if pd.isna(protocol_value) or not np.isfinite(float(protocol_value)):
            raise ValueError(f"CIC replay row {index} has an invalid protocol value")
        duration_microseconds = finite_nonnegative(row, ("Flow Duration",), index)
        flows.append(FlowTelemetry(
            timestamp="synthetic-replay",
            switch_id="synthetic-switch",
            host_id=f"synthetic-host-{index % 4 + 1}",
            flow_id=f"{id_prefix}-{index}",
            packet_count=int(finite_nonnegative(row, ("Tot Fwd Pkts", "Tot Bwd Pkts"), index)),
            byte_count=int(finite_nonnegative(row, ("TotLen Fwd Pkts", "TotLen Bwd Pkts"), index)),
            duration_seconds=duration_microseconds / 1_000_000.0,
            protocol=str(int(protocol_value)),
            controller_event="offline-cic-flow-replay",
            synthetic=True,
        ))
    return flows
