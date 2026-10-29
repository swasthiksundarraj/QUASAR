from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
import time
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MitigationAction:
    action_id: str
    action_type: str
    scope: str
    covers: tuple[str, ...]
    security_benefit: float
    availability_cost: float
    key_capacity_bits: float
    service_disruption: float
    mitigation_cost: float
    resource_group: str = "default"

    def __post_init__(self) -> None:
        if min(self.security_benefit, self.availability_cost, self.key_capacity_bits,
               self.service_disruption, self.mitigation_cost) < 0:
            raise ValueError("Action values must be nonnegative")


@dataclass
class QUBOProblem:
    variables: tuple[str, ...]
    coefficients: dict[tuple[int, int], float]
    constant: float
    actions: tuple[MitigationAction, ...]

    def energy(self, bits: tuple[int, ...] | list[int]) -> float:
        if len(bits) != len(self.variables) or any(bit not in (0, 1) for bit in bits):
            raise ValueError("Bit assignment does not match QUBO variables")
        return float(self.constant + sum(value * bits[i] * bits[j] for (i, j), value in self.coefficients.items()))

    def selected_actions(self, bits: tuple[int, ...] | list[int]) -> list[MitigationAction]:
        return [self.actions[i] for i, bit in enumerate(bits) if bit and i < len(self.actions)]


def build_mitigation_qubo(
    actions: list[MitigationAction],
    scenario_risks: dict[str, float],
    key_capacity_bits: float,
    max_disruption: float = 1.0,
    penalty: float = 100.0,
    key_capacity_cost: float = 0.1,
) -> QUBOProblem:
    """Build a one-choice QUBO; the final variable is the explicit no-op option."""
    if not actions:
        raise ValueError("At least one candidate action is required")
    if key_capacity_bits < 0 or max_disruption < 0 or penalty <= 0 or key_capacity_cost < 0:
        raise ValueError("Capacity/disruption must be nonnegative and penalty positive")
    names = [action.action_id for action in actions]
    if len(set(names)) != len(names) or "NOOP" in names:
        raise ValueError("Action IDs must be unique and cannot be NOOP")
    variables = tuple(names + ["NOOP"])
    size = len(variables)
    coefficients: dict[tuple[int, int], float] = {}
    constant = penalty
    # Penalty * (sum(x)-1)^2 makes the candidate action selection one-hot.
    for index, action in enumerate(actions):
        benefit = action.security_benefit * sum(
            min(1.0, max(0.0, float(scenario_risks.get(s, 0.0)))) for s in action.covers
        )
        key_cost = key_capacity_cost * action.key_capacity_bits / max(key_capacity_bits, 1.0)
        utility = (
            benefit - action.availability_cost - action.service_disruption
            - action.mitigation_cost - key_cost
        )
        coefficients[(index, index)] = -penalty - utility
        if action.key_capacity_bits > key_capacity_bits or action.service_disruption > max_disruption:
            coefficients[(index, index)] += penalty * 2
    noop_index = size - 1
    coefficients[(noop_index, noop_index)] = -penalty
    for left, right in product(range(size), repeat=2):
        if left < right:
            coefficients[(left, right)] = 2.0 * penalty
    return QUBOProblem(variables, coefficients, constant, tuple(actions))


def solve_exact(problem: QUBOProblem, max_variables: int = 20) -> dict[str, Any]:
    if len(problem.variables) > max_variables:
        raise ValueError(f"Exact solver limit is {max_variables} variables")
    started = time.perf_counter()
    best_bits: tuple[int, ...] | None = None
    best_energy = float("inf")
    for candidate in product((0, 1), repeat=len(problem.variables)):
        energy = problem.energy(candidate)
        if energy < best_energy:
            best_bits, best_energy = candidate, energy
    assert best_bits is not None
    return {
        "solver": "exact-enumeration",
        "bits": list(best_bits),
        "selected_action_ids": [a.action_id for a in problem.selected_actions(best_bits)],
        "objective": best_energy,
        "runtime_seconds": time.perf_counter() - started,
        "feasible_one_hot": sum(best_bits) == 1,
    }


def solve_qaoa(problem: QUBOProblem, reps: int = 1, seed: int = 42) -> dict[str, Any]:
    """Small-instance QAOA with Qiskit Statevector and a classical COBYLA outer loop."""
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import DiagonalGate
        from qiskit.quantum_info import Statevector
        from scipy.optimize import minimize
    except ImportError as exc:
        raise RuntimeError("Install Qiskit and SciPy to run the QAOA simulator") from exc
    n = len(problem.variables)
    if n > 12:
        raise ValueError("Statevector QAOA is limited to 12 variables")
    energies = np.asarray([
        problem.energy(tuple((index >> bit) & 1 for bit in range(n)))
        for index in range(2**n)
    ])

    def circuit_for(parameters: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(n)
        circuit.h(range(n))
        for layer in range(reps):
            gamma, beta = parameters[2 * layer : 2 * layer + 2]
            circuit.append(DiagonalGate(np.exp(-1j * gamma * energies)), range(n))
            for wire in range(n):
                circuit.rx(2.0 * beta, wire)
        return circuit

    def objective(parameters: np.ndarray) -> float:
        state = Statevector.from_instruction(circuit_for(parameters))
        return float(np.dot(np.abs(state.data) ** 2, energies))

    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    result = minimize(objective, rng.uniform(0, np.pi, size=2 * reps), method="COBYLA", options={"maxiter": 100})
    state = Statevector.from_instruction(circuit_for(result.x))
    sampled = rng.choice(2**n, p=np.abs(state.data) ** 2)
    bits = [(int(sampled) >> bit) & 1 for bit in range(n)]
    return {
        "solver": "qiskit-statevector-qaoa",
        "bits": bits,
        "selected_action_ids": [a.action_id for a in problem.selected_actions(bits)],
        "objective": problem.energy(bits),
        "expected_objective": float(result.fun),
        "runtime_seconds": time.perf_counter() - started,
        "feasible_one_hot": sum(bits) == 1,
        "optimizer_success": bool(result.success),
    }


def verify_plan(
    problem: QUBOProblem,
    bits: list[int],
    allowed_scopes: set[str],
    key_capacity_bits: float,
    max_disruption: float = 1.0,
) -> dict[str, Any]:
    reasons: list[str] = []
    if len(bits) != len(problem.variables) or any(bit not in (0, 1) for bit in bits):
        reasons.append("invalid-bit-vector")
        selected: list[MitigationAction] = []
    else:
        selected = problem.selected_actions(bits)
        if sum(bits) != 1:
            reasons.append("not-one-hot")
        if len(selected) > 1:
            reasons.append("multiple-mitigation-actions")
    for action in selected:
        if action.scope not in allowed_scopes:
            reasons.append(f"scope-not-allowlisted:{action.scope}")
        if action.key_capacity_bits > key_capacity_bits:
            reasons.append(f"key-capacity-exceeded:{action.action_id}")
        if action.service_disruption > max_disruption:
            reasons.append(f"service-impact-limit-exceeded:{action.action_id}")
        if action.action_type not in {"rate_limit", "quarantine_host", "protect_control_plane", "noop"}:
            reasons.append(f"action-type-not-allowlisted:{action.action_type}")
    accepted = not reasons
    return {
        "accepted": accepted,
        "reasons": reasons,
        "selected_action_ids": [a.action_id for a in selected] if accepted else ["NOOP"],
        "fallback": None if accepted else "NOOP",
    }


def apply_verified_plan(
    adapter: Any,
    problem: QUBOProblem,
    bits: list[int],
    allowed_scopes: set[str],
    key_capacity_bits: float,
    max_disruption: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    decision = verify_plan(problem, bits, allowed_scopes, key_capacity_bits, max_disruption)
    if not decision["accepted"]:
        return decision, {"status": "not-applied", "fallback": "NOOP"}
    actions = problem.selected_actions(bits)
    if not actions or actions[0].action_type == "noop":
        return decision, {"status": "simulated-noop"}
    result = adapter.apply(asdict(actions[0]))
    return decision, result
