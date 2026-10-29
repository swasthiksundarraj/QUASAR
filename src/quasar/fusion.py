from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from qkd.simulator import QKDTelemetry
from sdn.telemetry import FlowTelemetry


ATTACK_LIMITATIONS = {
    "A1": "Key-pool exhaustion is primarily detected from KMS accounting, not QATNet.",
    "A2": "Trusted-node compromise is inferred only from observable relay integrity events.",
    "A3": "Telemetry spoofing requires cross-checkable quality and consistency indicators.",
    "A4": "Physical-layer signatures are simulated QBER/rate anomalies, not physical measurements.",
    "A5": "Downgrade detection requires protocol-policy telemetry from the collector.",
    "A6": "Control-plane attacks require controller event counters; flow features alone are insufficient.",
    "A7": "Data-plane DoS/DDoS can use flow statistics; synthetic examples are not attack traffic.",
}


@dataclass(frozen=True)
class IncidentAssessment:
    risk_score: float
    risk_level: str
    scenario_scores: dict[str, float]
    evidence: tuple[str, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_incident(
    qkd_records: list[QKDTelemetry],
    flows: list[FlowTelemetry],
    qatnet_scores: dict[str, float] | None = None,
    controller_attack_events: int = 0,
) -> IncidentAssessment:
    evidence: list[str] = []
    scores = {f"A{i}": 0.0 for i in range(1, 8)}
    if qkd_records:
        for sample in qkd_records:
            if sample.allocation_failed or sample.key_pool_bits < 0.1 * max(sample.request_bits, 1):
                scores["A1"] = 0.95
                evidence.append("KMS allocation failure or near-empty key pool")
            elif sample.consumption_rate_bps > 1.5 * max(sample.generation_rate_bps, 1.0):
                scores["A1"] = max(scores["A1"], 0.8)
                evidence.append("KMS consumption rate substantially exceeds generation")
            if sample.relay_event:
                scores["A2"] = 0.9
                evidence.append(f"Trusted relay event: {sample.relay_event}")
            if not sample.quality.get("reported_generation_plausible", True) or not sample.quality.get("telemetry_complete", True):
                scores["A3"] = 0.85
                evidence.append("KMS telemetry quality/consistency check failed")
            if sample.qber > 0.10 or sample.secret_key_rate_bps < 0.25:
                scores["A4"] = 0.8
                evidence.append("Unusual simulated QBER or secret-key-rate observation")
            if sample.quality.get("forced_downgrade", False):
                scores["A5"] = 0.95
                evidence.append("Reported protocol/security downgrade")
    controller_signals = sum(
        any(term in flow.controller_event.lower() for term in ("unauthorized", "controller-attack", "reconnect-storm"))
        for flow in flows
    )
    controller_signals += max(0, controller_attack_events)
    if controller_signals > 0:
        scores["A6"] = min(0.5 + 0.1 * controller_signals, 1.0)
        evidence.append("Controller control-plane event threshold exceeded")
    if flows:
        candidates = [
            min(1.0, flow.packet_count / 50_000.0)
            for flow in flows
        ]
        if max(candidates) >= 0.8:
            scores["A7"] = max(candidates)
            evidence.append("High-volume flow counter anomaly")
    for scenario, score in (qatnet_scores or {}).items():
        if scenario == "A7":
            scores[scenario] = max(scores[scenario], min(max(float(score), 0.0), 1.0))
            evidence.append("QATNet flow-model attack probability contributed to A7 risk")
    overall = max(scores.values(), default=0.0)
    level = "high" if overall >= 0.8 else "medium" if overall >= 0.5 else "low"
    active = tuple(key for key, value in scores.items() if value > 0)
    return IncidentAssessment(
        risk_score=overall,
        risk_level=level,
        scenario_scores=scores,
        evidence=tuple(dict.fromkeys(evidence)),
        limitations=tuple(ATTACK_LIMITATIONS[key] for key in active),
    )
