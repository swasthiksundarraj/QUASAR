from __future__ import annotations

from typing import Any

import numpy as np


def qiskit_noisy_amplitude_features(
    vectors: np.ndarray,
    shots: int = 1024,
    depolarizing_probability: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    """Generate measured noisy amplitude-encoded features; this is not noisy QATNet."""
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error
    except ImportError as exc:
        raise RuntimeError("Install qiskit and qiskit-aer to use noisy-feature experiments") from exc
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2 or values.shape[1] & (values.shape[1] - 1):
        raise ValueError("Expected two-dimensional vectors with a power-of-two feature width")
    if shots < 1 or not 0 <= depolarizing_probability <= 1:
        raise ValueError("shots must be positive and error probability must be in [0, 1]")
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(depolarizing_probability, 1), ["id"])
    backend = AerSimulator(noise_model=noise, seed_simulator=seed)
    result = np.zeros_like(values)
    for row_index, vector in enumerate(values):
        norm = np.linalg.norm(vector)
        state = vector / norm if norm > 1e-12 else np.eye(1, len(vector), 0).ravel()
        qubits = int(np.log2(len(vector)))
        circuit = QuantumCircuit(qubits, qubits)
        circuit.initialize(state, range(qubits))
        circuit.barrier()
        for wire in range(qubits):
            circuit.id(wire)
        circuit.measure(range(qubits), range(qubits))
        compiled = transpile(circuit, backend)
        counts = backend.run(compiled, shots=shots).result().get_counts()
        for bitstring, count in counts.items():
            result[row_index, int(bitstring.replace(" ", ""), 2)] = count / shots
    return result.astype(np.float32)


def train_noisy_feature_learners(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    seed: int = 42,
) -> dict[str, Any]:
    """Separate classical learner/stacking experiment on noisy probability features."""
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.calibration import CalibratedClassifierCV

    learners = [
        ("logistic", LogisticRegression(max_iter=1000, random_state=seed)),
        ("random_forest", RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)),
        ("svc", CalibratedClassifierCV(estimator=SVC(), method="sigmoid", cv=3)),
    ]
    outputs: dict[str, Any] = {}
    for name, learner in learners:
        learner.fit(train_features, train_labels)
        outputs[name] = learner.predict_proba(test_features)
    stack = StackingClassifier(
        estimators=learners,
        final_estimator=LogisticRegression(max_iter=1000, random_state=seed),
        stack_method="predict_proba",
        n_jobs=1,
    )
    stack.fit(train_features, train_labels)
    outputs["stacked_ensemble"] = stack.predict_proba(test_features)
    return outputs
