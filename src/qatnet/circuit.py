from __future__ import annotations

import numpy as np


def amplitude_features(vectors: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """L2-normalize PCA vectors for amplitude encoding with a safe zero-vector path."""
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("vectors must be a two-dimensional array")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.maximum(norms, epsilon)
    zero_rows = norms[:, 0] <= epsilon
    normalized[zero_rows] = 0.0
    normalized[zero_rows, 0] = 1.0
    if values.shape[1] & (values.shape[1] - 1):
        raise ValueError("Amplitude-encoding dimensions must be a power of two")
    return normalized


def amplitude_probability_features(vectors: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Return computational-basis probabilities for amplitude-only features."""
    return np.square(amplitude_features(vectors, epsilon=epsilon))


class VariationalQuantumEncoder:
    """PennyLane state-vector feature encoder, pretrained then frozen."""

    def __init__(self, qubits: int, layers: int = 1, seed: int = 42) -> None:
        if qubits not in (4, 5, 6):
            raise ValueError("Supported qubit counts are 4, 5, and 6")
        if layers < 1:
            raise ValueError("layers must be positive")
        try:
            import pennylane as qml
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError("Install the research dependencies to use the PennyLane encoder") from exc
        self.qml = qml
        self.torch = torch
        self.qubits = qubits
        self.layers = layers
        self.device = qml.device("default.qubit", wires=qubits)

        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=range(qubits), normalize=True, pad_with=0.0)
            qml.StronglyEntanglingLayers(weights, wires=range(qubits))
            return qml.probs(wires=range(qubits))

        layer_shape = qml.StronglyEntanglingLayers.shape(n_layers=layers, n_wires=qubits)
        torch.manual_seed(seed)
        self.quantum_layer = qml.qnn.TorchLayer(circuit, {"weights": layer_shape})
        self._linear = nn.Linear
        self.probe = None

    def pretrain(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 15,
        batch_size: int = 256,
        learning_rate: float = 0.01,
        seed: int = 42,
    ) -> list[float]:
        torch = self.torch
        if epochs < 1 or batch_size < 1 or learning_rate <= 0:
            raise ValueError("epochs, batch_size and learning_rate must be positive")
        from torch.utils.data import DataLoader, TensorDataset
        features = amplitude_features(x)
        if features.shape[1] != 2**self.qubits:
            raise ValueError(f"Expected {2**self.qubits} PCA dimensions for Q={self.qubits}")
        y = np.asarray(y, dtype=np.int64)
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2:
            raise ValueError("VQC pretraining requires at least two classes")
        if self.probe is None or self.probe.out_features != len(unique):
            self.probe = self._linear(2**self.qubits, len(unique))
        # A balanced subset is sampled without replacement; caller can provide a prebalanced subset.
        rng = np.random.default_rng(seed)
        selected = np.concatenate([
            rng.choice(np.flatnonzero(y == label), size=int(counts.min()), replace=False)
            for label in unique
        ])
        rng.shuffle(selected)
        dataset = TensorDataset(
            torch.tensor(features[selected], dtype=torch.float32),
            torch.tensor(y[selected], dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
        optimizer = torch.optim.Adam(
            list(self.quantum_layer.parameters()) + list(self.probe.parameters()),
            lr=learning_rate,
        )
        losses: list[float] = []
        self.quantum_layer.train()
        self.probe.train()
        for _ in range(epochs):
            total_loss = 0.0
            samples = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                probabilities = self.quantum_layer(batch_x)
                logits = self.probe(probabilities)
                loss = torch.nn.functional.cross_entropy(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(batch_y)
                samples += len(batch_y)
            losses.append(total_loss / samples)
        for parameter in self.quantum_layer.parameters():
            parameter.requires_grad_(False)
        self.quantum_layer.eval()
        return losses

    def transform(self, x: np.ndarray, batch_size: int = 128) -> np.ndarray:
        torch = self.torch
        features = amplitude_features(x)
        if features.shape[1] != 2**self.qubits:
            raise ValueError(f"Expected {2**self.qubits} PCA dimensions for Q={self.qubits}")
        output = []
        self.quantum_layer.eval()
        with torch.no_grad():
            for start in range(0, len(features), batch_size):
                batch = torch.tensor(features[start : start + batch_size], dtype=torch.float32)
                output.append(self.quantum_layer(batch).cpu().numpy())
        return np.concatenate(output, axis=0) if output else np.empty((0, 2**self.qubits), dtype=np.float32)
