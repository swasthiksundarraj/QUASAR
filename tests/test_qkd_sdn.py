from datetime import datetime, timezone

import pytest

from qkd.simulator import QKDConfig, QKDSimulator
from sdn.mininet_adapter import MininetStatsAdapter
from quasar.modules import (
    check_qkd_consistency,
    estimate_key_depletion,
    qatnet_flow_detection,
    score_trusted_nodes,
    TelemetryCollector,
)
from sdn.telemetry import FlowTelemetry, MockSDNAdapter


def test_key_pool_reservation_and_exhaustion():
    simulator = QKDSimulator(QKDConfig(
        interval_seconds=2, initial_key_pool_bits=100,
        generation_rate_bps=10, consumption_rate_bps=20, seed=1,
    ))
    normal = simulator.step(timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc))
    assert normal.key_pool_bits == 80
    assert normal.allocated_bits == 40
    exhausted = simulator.step("key_pool_exhaustion")
    assert exhausted.allocation_failed
    assert exhausted.key_pool_bits == 0


def test_scenario_markers_and_deterministic_seed():
    first = QKDSimulator(QKDConfig(seed=8)).step("unusual_channel")
    second = QKDSimulator(QKDConfig(seed=8)).step("unusual_channel")
    assert first.qber == second.qber
    assert first.qber > 0.1
    assert first.secret_key_rate_bps < 500
    assert QKDSimulator(QKDConfig()).step("telemetry_spoofing").quality["reported_generation_plausible"] is False


def test_mock_sdn_is_local_and_validates_counters():
    flows = MockSDNAdapter(seed=5).collect(count=3, anomaly=True)
    assert len(flows) == 3
    assert flows[-1].synthetic
    assert flows[-1].controller_event == "synthetic-flow-anomaly"
    with pytest.raises(ValueError):
        FlowTelemetry("now", "s1", "h1", "f1", -1, 0, 0, "TCP")


def test_qkd_quality_depletion_and_relay_trust_are_observable_only():
    simulator = QKDSimulator(QKDConfig(initial_key_pool_bits=100, seed=10))
    first = simulator.step("normal", timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc))
    second = simulator.step("trusted_node_anomaly", timestamp=datetime(2025, 1, 1, 0, 0, 1, tzinfo=timezone.utc))
    third = simulator.step("suspicious_consumption", timestamp=datetime(2025, 1, 1, 0, 0, 2, tzinfo=timezone.utc))
    assert check_qkd_consistency(first).consistent
    assert estimate_key_depletion([first, second, third])["estimated_seconds_to_empty"] is not None
    trust = score_trusted_nodes([first, second, third])
    assert trust["trust_score"] < 1
    assert trust["evidence_scope"] == "observable simulator relay events only"


def test_flow_detection_adapter_requires_explicit_feature_mapping():
    class Model:
        def predict_proba(self, frame):
            import numpy as np
            assert list(frame.columns) == ["Total Fwd Packets"]
            return np.array([[0.1, 0.9]])

    flow = MockSDNAdapter(seed=5).collect(count=1)[0]
    result = qatnet_flow_detection(
        Model(), [flow], {"Total Fwd Packets": "packet_count"}, ("BENIGN", "Attack"), {"Attack"}
    )
    assert result[0]["malicious"]
    with pytest.raises(ValueError, match="unavailable"):
        qatnet_flow_detection(
            Model(), [flow], {"Total Fwd Packets": "missing_feature"},
            ("BENIGN", "Attack"), {"Attack"},
        )


def test_telemetry_collector_combines_synthetic_qkd_and_sdn():
    collector = TelemetryCollector(
        MockSDNAdapter(seed=1), QKDSimulator(QKDConfig(seed=1))
    )
    payload = collector.collect(scenario="trusted_node_anomaly", flow_count=2, flow_anomaly=True)
    assert payload["qkd"].relay_event == "relay-integrity-anomaly"
    assert len(payload["flows"]) == 2
    assert payload["synthetic_qkd"] and payload["synthetic_flows"]


def test_mininet_adapter_refuses_non_linux(monkeypatch):
    import sdn.mininet_adapter as adapter_module
    monkeypatch.setattr(adapter_module.platform, "system", lambda: "Darwin")
    with pytest.raises(RuntimeError, match="isolated Linux VM"):
        MininetStatsAdapter().start()
