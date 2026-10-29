from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from qatnet.service import QATNetDetectionService
from qkd.simulator import QKDConfig, QKDSimulator
from quasar.audit import AuditLog
from quasar.fusion import assess_incident
from quasar.qubo import (
    MitigationAction,
    apply_verified_plan,
    build_mitigation_qubo,
    solve_exact,
)
from sdn.telemetry import FlowTelemetry, MockSDNAdapter


def run_qatnet_replay(
    service: QATNetDetectionService,
    flows: Sequence[FlowTelemetry],
    flow_features: pd.DataFrame,
    output_dir: str | Path,
    seed: int = 42,
) -> dict:
    if not flows:
        raise ValueError("At least one flow is required for a QATNet replay")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)

    detections = service.predict(flows, flow_features)
    risk = max(result.attack_probability for result in detections)
    qkd_record = QKDSimulator(QKDConfig(seed=seed)).step()
    assessment = assess_incident(
        [qkd_record], list(flows), qatnet_scores={"A7": risk}
    )
    target = max(detections, key=lambda result: result.attack_probability)
    target_flow = next(flow for flow in flows if flow.flow_id == target.flow_id)
    action = MitigationAction(
        "simulated-rate-limit",
        "rate_limit",
        target_flow.host_id,
        ("A7",),
        security_benefit=1.0,
        availability_cost=0.05,
        key_capacity_bits=0,
        service_disruption=0.05,
        mitigation_cost=0.05,
    )
    problem = build_mitigation_qubo(
        [action], {"A7": risk}, key_capacity_bits=1_000
    )
    plan = solve_exact(problem)
    adapter = MockSDNAdapter(seed=seed)
    allowed_scopes = {flow.host_id for flow in flows}
    decision, mitigation = apply_verified_plan(
        adapter, problem, plan["bits"], allowed_scopes, key_capacity_bits=1_000
    )
    rollback = None
    if mitigation.get("status") == "simulated":
        rollback = adapter.rollback(str(mitigation["action"]["action_id"]))

    audit = AuditLog(output / "audit.sqlite3")
    audit.record("qatnet-flow-detections", {
        "model_version": service.model_version,
        "detections": [result.to_dict() for result in detections],
    })
    audit.record("cross-layer-assessment", assessment.to_dict())
    audit.record("mitigation-verification", decision)
    audit.record("mitigation-result", mitigation)
    if rollback is not None:
        audit.record("mitigation-rollback", rollback)
    result = {
        "mode": "offline-cic-flow-replay",
        "live_network_actions": False,
        "qkd_is_synthetic": True,
        "sdn_flow_metadata_is_synthetic": all(flow.synthetic for flow in flows),
        "model_version": service.model_version,
        "model_classes": list(service.class_names),
        "flows": [flow.to_dict() for flow in flows],
        "detections": [item.to_dict() for item in detections],
        "qkd": asdict(qkd_record),
        "assessment": assessment.to_dict(),
        "mitigation_candidate": asdict(action),
        "solver": plan,
        "verification": decision,
        "mitigation": mitigation,
        "rollback": rollback,
        "mitigation_active_after_rollback": bool(adapter.applied_actions),
    }
    (output / "replay.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result
