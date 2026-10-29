from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Any, Callable

import numpy as np
import pandas as pd

from qkd.simulator import QKDConfig, QKDSimulator, QKDTelemetry
from sdn.telemetry import FlowTelemetry, MockSDNAdapter


@dataclass(frozen=True)
class ConsistencyResult:
    consistent: bool
    indicators: tuple[str, ...]
    note: str = "QBER/rate anomalies are simulated consistency indicators, not proof of eavesdropping."


def check_qkd_consistency(record: QKDTelemetry, generation_tolerance: float = 1.5) -> ConsistencyResult:
    indicators: list[str] = []
    if not 0 <= record.qber <= 0.5:
        indicators.append("qber-out-of-range")
    if min(record.secret_key_rate_bps, record.generation_rate_bps, record.consumption_rate_bps) < 0:
        indicators.append("negative-rate")
    if record.secret_key_rate_bps > record.generation_rate_bps * generation_tolerance:
        indicators.append("secret-rate-exceeds-generation")
    if not record.quality.get("telemetry_complete", False):
        indicators.append("incomplete-telemetry")
    if not record.quality.get("reported_generation_plausible", False):
        indicators.append("generation-rate-outlier")
    if record.quality.get("forced_downgrade", False):
        indicators.append("security-downgrade-reported")
    return ConsistencyResult(not indicators, tuple(indicators))


def estimate_key_depletion(records: list[QKDTelemetry]) -> dict[str, Any]:
    if not records:
        raise ValueError("At least one KMS telemetry sample is required")
    latest = records[-1]
    if len(records) > 1:
        previous = records[-2]
        try:
            elapsed = max(0.0, (
                pd.Timestamp(latest.timestamp) - pd.Timestamp(previous.timestamp)
            ).total_seconds())
        except (TypeError, ValueError):
            elapsed = 0.0
        observed_rate = (
            (previous.key_pool_bits - latest.key_pool_bits) / elapsed if elapsed > 0 else
            latest.consumption_rate_bps - latest.generation_rate_bps
        )
    else:
        observed_rate = latest.consumption_rate_bps - latest.generation_rate_bps
    seconds = latest.key_pool_bits / observed_rate if observed_rate > 0 else None
    return {
        "key_pool_bits": latest.key_pool_bits,
        "net_depletion_rate_bps": max(0.0, float(observed_rate)),
        "estimated_seconds_to_empty": max(0.0, float(seconds)) if seconds is not None else None,
        "allocation_failed": latest.allocation_failed,
        "synthetic": True,
    }


def score_trusted_nodes(records: list[QKDTelemetry]) -> dict[str, Any]:
    events = [record.relay_event for record in records if record.relay_event]
    # This score reflects only relay events visible to the simulator.
    trust = exp(-0.7 * len(events))
    return {
        "trust_score": float(trust),
        "observable_relay_events": events,
        "evidence_scope": "observable simulator relay events only",
    }


class TelemetryCollector:
    def __init__(self, sdn_adapter: MockSDNAdapter, qkd_simulator: QKDSimulator | None = None) -> None:
        self.sdn_adapter = sdn_adapter
        self.qkd_simulator = qkd_simulator

    def collect_flows(self, count: int = 10, anomaly: bool = False) -> list[FlowTelemetry]:
        return self.sdn_adapter.collect(count, anomaly)

    def collect(self, scenario: str = "normal", flow_count: int = 10, flow_anomaly: bool = False) -> dict[str, Any]:
        if self.qkd_simulator is None:
            raise RuntimeError("A QKD simulator/collector must be configured")
        return {
            "qkd": self.qkd_simulator.step(scenario),
            "flows": self.collect_flows(flow_count, flow_anomaly),
            "synthetic_qkd": True,
            "synthetic_flows": isinstance(self.sdn_adapter, MockSDNAdapter),
        }


def qatnet_flow_detection(
    model: Any,
    flows: list[FlowTelemetry],
    feature_mapping: dict[str, str],
    class_names: tuple[str, ...],
    malicious_labels: set[str],
) -> list[dict[str, Any]]:
    if not flows:
        return []
    frame = pd.DataFrame([flow.to_dict() for flow in flows])
    missing = set(feature_mapping.values()) - set(frame.columns)
    if missing:
        raise ValueError(f"Flow feature mapping refers to unavailable telemetry fields: {sorted(missing)}")
    model_features = pd.DataFrame({
        model_name: frame[telemetry_name]
        for model_name, telemetry_name in feature_mapping.items()
    })
    probabilities = model.predict_proba(model_features)
    if probabilities.shape[1] != len(class_names):
        raise ValueError("Class-name mapping does not match the model output dimension")
    result = []
    for flow, probs in zip(flows, probabilities):
        class_index = int(np.argmax(probs))
        predicted = class_names[class_index]
        result.append({
            "flow_id": flow.flow_id,
            "predicted_label": predicted,
            "confidence": float(probs[class_index]),
            "malicious": predicted in malicious_labels,
        })
    return result
