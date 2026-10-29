from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from typing import Any


@dataclass(frozen=True)
class FlowTelemetry:
    timestamp: str
    switch_id: str
    host_id: str
    flow_id: str
    packet_count: int
    byte_count: int
    duration_seconds: float
    protocol: str
    controller_event: str = "flow-stats"
    detection_result: str = "unknown"
    synthetic: bool = True

    def __post_init__(self) -> None:
        if (
            self.packet_count < 0 or self.byte_count < 0
            or not math.isfinite(self.duration_seconds) or self.duration_seconds < 0
        ):
            raise ValueError("Flow counters and duration must be finite and nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MockSDNAdapter:
    """Deterministic local telemetry source; never sends network traffic."""

    def __init__(self, seed: int = 42) -> None:
        import numpy as np
        self.rng = np.random.default_rng(seed)
        self.applied_actions: list[dict[str, Any]] = []
        self.rollback_history: list[dict[str, Any]] = []

    def collect(self, count: int = 10, anomaly: bool = False) -> list[FlowTelemetry]:
        if count < 0:
            raise ValueError("count cannot be negative")
        now = datetime.now(timezone.utc).isoformat()
        records = []
        for index in range(count):
            attack_like = anomaly and index == count - 1
            packets = int(self.rng.integers(10, 500) if not attack_like else self.rng.integers(50_000, 100_000))
            bytes_count = int(packets * self.rng.integers(64, 1400))
            records.append(FlowTelemetry(
                timestamp=now, switch_id="s1", host_id=f"h{index % 4 + 1}",
                flow_id=f"mock-{index}", packet_count=packets,
                byte_count=bytes_count, duration_seconds=float(self.rng.uniform(0.1, 60.0)),
                protocol=str(self.rng.choice(["TCP", "UDP", "ICMP"])),
                controller_event="synthetic-flow-anomaly" if attack_like else "flow-stats",
                detection_result="candidate-anomaly" if attack_like else "unknown",
            ))
        return records

    def apply(self, action: dict[str, Any]) -> dict[str, Any]:
        result = {
            "status": "simulated",
            "action": dict(action),
            "scope": action.get("scope"),
            "service_impact": float(action.get("service_disruption", 0.0)),
        }
        self.applied_actions.append(result)
        return result

    def rollback(self, action_id: str) -> dict[str, Any]:
        existing_count = len(self.applied_actions)
        self.applied_actions = [
            item for item in self.applied_actions
            if item.get("action", {}).get("action_id") != action_id
        ]
        if len(self.applied_actions) == existing_count:
            raise KeyError(f"No simulated mitigation is active for action {action_id!r}")
        result = {"status": "simulated-rollback", "action_id": action_id}
        self.rollback_history.append(result)
        return result


def start_mininet_lab() -> Any:
    from sdn.mininet_adapter import MininetStatsAdapter
    adapter = MininetStatsAdapter()
    adapter.start()
    return adapter
