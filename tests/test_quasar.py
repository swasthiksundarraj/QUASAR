from quasar.fusion import assess_incident
from quasar.qubo import (
    MitigationAction,
    apply_verified_plan,
    build_mitigation_qubo,
    solve_exact,
    verify_plan,
)
from sdn.telemetry import MockSDNAdapter
from qkd.simulator import QKDConfig, QKDSimulator


def actions():
    return [
        MitigationAction("rate", "rate_limit", "h1", ("A7",), 1.0, 0.1, 0, 0.1, 0.1),
        MitigationAction("quarantine", "quarantine_host", "h2", ("A7",), 0.8, 0.4, 0, 0.5, 0.2),
    ]


def test_exact_qubo_finds_best_feasible_one_hot_action():
    problem = build_mitigation_qubo(actions(), {"A7": 1.0}, key_capacity_bits=100)
    result = solve_exact(problem)
    assert result["feasible_one_hot"]
    assert result["selected_action_ids"] == ["rate"]
    assert abs(result["objective"] - (-0.7)) < 1e-8


def test_verifier_rejects_untrusted_scope_and_uses_noop():
    problem = build_mitigation_qubo(actions(), {"A7": 1.0}, key_capacity_bits=100)
    unsafe = [0, 1, 0]
    decision = verify_plan(problem, unsafe, {"h1"}, key_capacity_bits=100)
    assert not decision["accepted"]
    assert decision["selected_action_ids"] == ["NOOP"]
    decision, mitigation = apply_verified_plan(
        MockSDNAdapter(), problem, unsafe, {"h1"}, key_capacity_bits=100
    )
    assert not decision["accepted"]
    assert mitigation["status"] == "not-applied"


def test_mock_mitigation_can_be_rolled_back():
    adapter = MockSDNAdapter()
    applied = adapter.apply({"action_id": "temporary", "scope": "h1"})
    assert applied["service_impact"] == 0
    rollback = adapter.rollback("temporary")
    assert rollback["status"] == "simulated-rollback"
    assert adapter.applied_actions == []


def test_cross_layer_fusion_marks_qkd_exhaustion_and_flow_anomaly():
    simulator = QKDSimulator(QKDConfig(initial_key_pool_bits=20, seed=4))
    exhausted = simulator.step("key_pool_exhaustion")
    channel = simulator.step("unusual_channel")
    flows = MockSDNAdapter(seed=4).collect(2, anomaly=True)
    result = assess_incident([exhausted, channel], flows)
    assert result.scenario_scores["A1"] > 0.8
    assert result.scenario_scores["A7"] >= 0.8
    assert "not physical measurements" in " ".join(result.limitations)


def test_suspicious_key_consumption_and_controller_event_are_separate_signals():
    qkd = QKDSimulator(QKDConfig(initial_key_pool_bits=100_000, seed=4)).step("suspicious_consumption")
    flows = MockSDNAdapter(seed=5).collect(count=1)
    from dataclasses import replace
    flows = [replace(flows[0], controller_event="unauthorized-controller-event")]
    result = assess_incident([qkd], flows)
    assert result.scenario_scores["A1"] >= 0.8
    assert result.scenario_scores["A6"] >= 0.5
