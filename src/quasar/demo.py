from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

from qkd.simulator import QKDConfig, QKDSimulator
from quasar.audit import AuditLog
from quasar.fusion import assess_incident
from quasar.qubo import (
    MitigationAction,
    apply_verified_plan,
    build_mitigation_qubo,
    solve_exact,
    solve_qaoa,
    verify_plan,
)
from sdn.telemetry import MockSDNAdapter


def run_demo(output_dir: str | Path = "outputs/demo") -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    qkd = QKDSimulator(QKDConfig(initial_key_pool_bits=100.0, seed=7))
    telemetry = qkd.step("key_pool_exhaustion")
    sdn = MockSDNAdapter(seed=7)
    flows = sdn.collect(4, anomaly=True)
    assessment = assess_incident([telemetry], flows)
    actions = [
        MitigationAction(
            "limit-host", "rate_limit", "h4", ("A1", "A7"),
            security_benefit=1.0, availability_cost=0.1, key_capacity_bits=0,
            service_disruption=0.1, mitigation_cost=0.1,
        ),
        MitigationAction(
            "protect-kms", "protect_control_plane", "kms-service", ("A1", "A6"),
            security_benefit=1.0, availability_cost=0.2, key_capacity_bits=0,
            service_disruption=0.2, mitigation_cost=0.2,
        ),
    ]
    risks = {"A1": assessment.scenario_scores["A1"], "A7": assessment.scenario_scores["A7"], "A6": 0.0}
    problem = build_mitigation_qubo(actions, risks, key_capacity_bits=10_000)
    classical = solve_exact(problem)
    try:
        qaoa = solve_qaoa(problem, reps=1)
    except (RuntimeError, ValueError) as exc:
        qaoa = {"status": "unavailable", "reason": str(exc)}
    response_deadline_seconds = 1.0
    classical["deadline_compliant"] = classical["runtime_seconds"] <= response_deadline_seconds
    classical_verifier = verify_plan(problem, classical["bits"], {"h4", "kms-service"}, 10_000)
    if "bits" in qaoa:
        qaoa["deadline_compliant"] = qaoa["runtime_seconds"] <= response_deadline_seconds
        qaoa_verifier = verify_plan(problem, qaoa["bits"], {"h4", "kms-service"}, 10_000)
    else:
        qaoa_verifier = {"accepted": False, "reasons": ["solver-unavailable"], "fallback": "NOOP"}
    plan = classical["bits"]
    decision, mitigation = apply_verified_plan(
        sdn, problem, plan, {"h4", "kms-service"}, key_capacity_bits=10_000
    )
    rollback = None
    if decision["accepted"] and mitigation.get("status") == "simulated":
        rollback = sdn.rollback(str(mitigation["action"]["action_id"]))
    audit = AuditLog(output / "audit.sqlite3")
    audit.record("incident-assessment", assessment.to_dict())
    audit.record("mitigation-verification", decision)
    audit.record("mitigation-result", mitigation)
    if rollback is not None:
        audit.record("mitigation-rollback", rollback)
    qkd.export(output / "qkd-telemetry.csv", output / "qkd-telemetry.json")
    result = {
        "synthetic_telemetry": True,
        "simulated_mitigation": True,
        "qkd": asdict(telemetry),
        "flow_telemetry": [flow.to_dict() for flow in flows],
        "assessment": assessment.to_dict(),
        "classical_solver": classical,
        "qaoa_solver": qaoa,
        "solver_comparison": {
            "response_deadline_seconds": response_deadline_seconds,
            "classical_feasible": classical_verifier["accepted"],
            "qaoa_feasible": qaoa_verifier["accepted"],
            "classical_verifier": classical_verifier,
            "qaoa_verifier": qaoa_verifier,
            "simulated_mitigation_success": decision["accepted"] and mitigation.get("status") == "simulated",
            "simulated_service_impact": mitigation.get("service_impact", 0.0),
            "mitigation_active_after_rollback": bool(sdn.applied_actions),
        },
        "candidate_mitigations": [asdict(action) for action in actions],
        "verifier": decision,
        "mitigation": mitigation,
        "rollback": rollback,
        "rollback_history": sdn.rollback_history,
    }
    (output / "demo.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result
