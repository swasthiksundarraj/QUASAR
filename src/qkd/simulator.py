from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QKDConfig:
    interval_seconds: float = 1.0
    initial_key_pool_bits: float = 100_000.0
    generation_rate_bps: float = 2_000.0
    consumption_rate_bps: float = 1_000.0
    secret_key_rate_bps: float = 1_800.0
    qber_mean: float = 0.02
    qber_std: float = 0.004
    seed: int = 42

    def __post_init__(self) -> None:
        values = (
            self.interval_seconds, self.initial_key_pool_bits, self.generation_rate_bps,
            self.consumption_rate_bps, self.secret_key_rate_bps, self.qber_mean, self.qber_std,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("QKD configuration values must be finite")
        if self.interval_seconds <= 0 or self.initial_key_pool_bits < 0:
            raise ValueError("Interval must be positive and initial key pool cannot be negative")
        if min(self.generation_rate_bps, self.consumption_rate_bps, self.secret_key_rate_bps, self.qber_std) < 0:
            raise ValueError("Rates and QBER standard deviation must be nonnegative")
        if not 0 <= self.qber_mean <= 0.5:
            raise ValueError("qber_mean must be in [0, 0.5]")


@dataclass
class QKDTelemetry:
    timestamp: str
    qber: float
    secret_key_rate_bps: float
    generation_rate_bps: float
    consumption_rate_bps: float
    key_pool_bits: float
    request_bits: float
    allocated_bits: float
    allocation_failed: bool
    relay_event: str | None
    scenario: str
    quality: dict[str, bool] = field(default_factory=dict)


class QKDSimulator:
    SCENARIOS = {
        "normal", "key_pool_exhaustion", "suspicious_consumption",
        "telemetry_spoofing", "unusual_channel", "forced_downgrade",
        "trusted_node_anomaly",
    }

    def __init__(self, config: QKDConfig = QKDConfig()) -> None:
        self.config = config
        self.key_pool_bits = float(config.initial_key_pool_bits)
        self.rng = np.random.default_rng(config.seed)
        self.records: list[QKDTelemetry] = []

    def step(
        self,
        scenario: str = "normal",
        timestamp: datetime | None = None,
        request_bits: float | None = None,
    ) -> QKDTelemetry:
        if scenario not in self.SCENARIOS:
            raise ValueError(f"Unknown QKD scenario: {scenario}")
        if request_bits is not None and (not math.isfinite(request_bits) or request_bits < 0):
            raise ValueError("request_bits must be finite and nonnegative")
        cfg = self.config
        stamp = timestamp or datetime.now(timezone.utc)
        generation = cfg.generation_rate_bps
        consumption = cfg.consumption_rate_bps
        secret_rate = cfg.secret_key_rate_bps
        qber = float(np.clip(self.rng.normal(cfg.qber_mean, cfg.qber_std), 0.0, 0.5))
        relay: str | None = None
        request = request_bits if request_bits is not None else consumption * cfg.interval_seconds

        if scenario == "key_pool_exhaustion":
            request = max(request, self.key_pool_bits + generation * cfg.interval_seconds * 2)
        elif scenario == "suspicious_consumption":
            consumption *= 8
            request = max(request, consumption * cfg.interval_seconds)
        elif scenario == "telemetry_spoofing":
            generation *= 3
        elif scenario == "unusual_channel":
            qber = float(np.clip(qber + 0.20, 0.0, 0.5))
            secret_rate *= 0.1
        elif scenario == "forced_downgrade":
            secret_rate *= 0.1
        elif scenario == "trusted_node_anomaly":
            relay = "relay-integrity-anomaly"

        available = max(0.0, self.key_pool_bits + generation * cfg.interval_seconds)
        allocation = min(request, available)
        failed = allocation < request
        self.key_pool_bits = max(0.0, available - allocation)
        quality = {
            "qber_in_range": 0.0 <= qber <= 0.5,
            "rates_nonnegative": all(rate >= 0 for rate in (generation, consumption, secret_rate)),
            "reported_generation_plausible": generation <= cfg.generation_rate_bps * 1.5,
            "telemetry_complete": True,
            "forced_downgrade": scenario == "forced_downgrade",
        }
        record = QKDTelemetry(
            timestamp=stamp.isoformat(),
            qber=qber,
            secret_key_rate_bps=secret_rate,
            generation_rate_bps=generation,
            consumption_rate_bps=consumption,
            key_pool_bits=self.key_pool_bits,
            request_bits=request,
            allocated_bits=allocation,
            allocation_failed=failed,
            relay_event=relay,
            scenario=scenario,
            quality=quality,
        )
        self.records.append(record)
        return record

    def run(self, steps: int, scenario: str = "normal", start: datetime | None = None) -> list[QKDTelemetry]:
        if steps < 0:
            raise ValueError("steps cannot be negative")
        base = start or datetime.now(timezone.utc)
        for index in range(steps):
            self.step(
                scenario=scenario,
                timestamp=base + timedelta(seconds=index * self.config.interval_seconds),
            )
        return self.records[-steps:] if steps else []

    def export(self, csv_path: str | Path, json_path: str | Path | None = None) -> None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [asdict(record) for record in self.records]
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        if json_path is not None:
            path = Path(json_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rows, indent=2) + "\n")
