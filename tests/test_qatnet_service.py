import numpy as np
import pandas as pd
import pytest

from qatnet.service import QATNetDetectionService, telemetry_from_cic_records
from quasar.qatnet_demo import run_qatnet_replay
from sdn.telemetry import FlowTelemetry


class FittedPreprocessor:
    input_feature_names = ("feature_a", "feature_b")


class StubModel:
    preprocessor = FittedPreprocessor()

    def predict_proba(self, features, use_cache=False):
        assert list(features.columns) == ["feature_a", "feature_b"]
        return np.array([[0.05, 0.95], [0.8, 0.2]], dtype=np.float64)


def sample_flows():
    return [
        FlowTelemetry("offline", "s1", "h1", "f1", 10, 100, 1.0, "TCP"),
        FlowTelemetry("offline", "s1", "h2", "f2", 20, 200, 2.0, "UDP"),
    ]


def test_detection_service_returns_class_probabilities_and_version():
    service = QATNetDetectionService(
        StubModel(), ("Benign", "Bot"), {"Bot"}, "model-123"
    )
    results = service.predict(
        sample_flows(),
        pd.DataFrame({"feature_a": [1, 2], "feature_b": [3, 4], "Label": ["Bot", "Benign"]}),
    )

    assert results[0].predicted_class == "Bot"
    assert results[0].attack_probability == pytest.approx(0.95)
    assert results[0].probabilities == {"Benign": 0.05, "Bot": 0.95}
    assert results[0].model_version == "model-123"
    assert results[1].predicted_class == "Benign"


def test_detection_service_fails_closed_for_incomplete_or_misaligned_inputs():
    service = QATNetDetectionService(
        StubModel(), ("Benign", "Bot"), {"Bot"}, "model-123"
    )
    with pytest.raises(ValueError, match="missing model features"):
        service.predict(sample_flows(), pd.DataFrame({"feature_a": [1, 2]}))
    with pytest.raises(ValueError, match="exactly one aligned"):
        service.predict(sample_flows()[:1], pd.DataFrame({
            "feature_a": [1, 2], "feature_b": [3, 4],
        }))


def test_end_to_end_detection_fusion_and_verified_mock_rollback(tmp_path):
    service = QATNetDetectionService(
        StubModel(), ("Benign", "Bot"), {"Bot"}, "model-123"
    )
    result = run_qatnet_replay(
        service,
        sample_flows(),
        pd.DataFrame({"feature_a": [1, 2], "feature_b": [3, 4]}),
        tmp_path / "run",
        seed=3,
    )

    assert result["live_network_actions"] is False
    assert result["assessment"]["scenario_scores"]["A7"] == pytest.approx(0.95)
    assert result["verification"]["accepted"]
    assert result["mitigation"]["status"] == "simulated"
    assert result["rollback"]["status"] == "simulated-rollback"
    assert not result["mitigation_active_after_rollback"]
    assert (tmp_path / "run" / "replay.json").is_file()


def test_cic_replay_telemetry_requires_valid_flow_fields():
    records = pd.DataFrame({
        "Tot Fwd Pkts": [3],
        "Tot Bwd Pkts": [2],
        "TotLen Fwd Pkts": [100],
        "TotLen Bwd Pkts": [150],
        "Flow Duration": [2_000_000.0],
        "Protocol": [6],
    })
    flow = telemetry_from_cic_records(records)[0]
    assert flow.packet_count == 5
    assert flow.byte_count == 250
    assert flow.duration_seconds == 2
    assert flow.synthetic

    records.loc[0, "Flow Duration"] = np.inf
    with pytest.raises(ValueError, match="invalid telemetry values"):
        telemetry_from_cic_records(records)
