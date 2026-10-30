from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import time
from typing import Any, Callable, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from data.dataset import DataSplits
from qatnet.circuit import (
    VariationalQuantumEncoder,
    amplitude_features,
    amplitude_probability_features,
)


@dataclass(frozen=True)
class TrainingConfig:
    qubits: int = 4
    vqc_layers: int = 1
    pretrain_samples: int = 3000
    pretrain_epochs: int = 15
    batch_size: int = 256
    learning_rate: float = 0.01
    tabnet_max_epochs: int = 100
    tabnet_patience: int = 15
    seed: int = 42

    def __post_init__(self) -> None:
        if self.qubits not in (4, 5, 6):
            raise ValueError("qubits must be 4, 5, or 6")
        if min(self.vqc_layers, self.pretrain_samples, self.pretrain_epochs,
               self.batch_size, self.tabnet_max_epochs, self.tabnet_patience) < 1:
            raise ValueError("Layer, sample, epoch, batch, and patience settings must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


@dataclass(frozen=True)
class TabNetCandidate:
    n_d: int
    n_a: int
    n_steps: int
    mask_type: str

    def __post_init__(self) -> None:
        if min(self.n_d, self.n_a, self.n_steps) < 1:
            raise ValueError("TabNet candidate dimensions and steps must be positive")
        if self.mask_type not in {"sparsemax", "entmax"}:
            raise ValueError("mask_type must be 'sparsemax' or 'entmax'")


@dataclass(frozen=True)
class PaperTabNetProfile:
    """Explicit paper-profile settings; candidate grid is an implementation assumption."""

    max_epochs: int = 60
    patience: int = 15
    batch_size: int = 4096
    virtual_batch_size: int = 512
    selection_seed: int = 101
    model_seeds: tuple[int, ...] = (101, 202, 402)
    lambda_sparse: float = 0.001
    device_name: str = "auto"
    candidates: tuple[TabNetCandidate, ...] = (
        TabNetCandidate(n_d=8, n_a=8, n_steps=3, mask_type="sparsemax"),
        TabNetCandidate(n_d=16, n_a=16, n_steps=3, mask_type="sparsemax"),
        TabNetCandidate(n_d=8, n_a=8, n_steps=3, mask_type="entmax"),
    )

    def __post_init__(self) -> None:
        if min(self.max_epochs, self.patience, self.batch_size, self.virtual_batch_size) < 1:
            raise ValueError("Paper TabNet epoch, patience, and batch settings must be positive")
        if self.virtual_batch_size >= self.batch_size:
            raise ValueError("virtual_batch_size must be smaller than batch_size")
        if not self.model_seeds or len(set(self.model_seeds)) != len(self.model_seeds):
            raise ValueError("model_seeds must be a non-empty sequence of unique seeds")
        if self.selection_seed not in self.model_seeds:
            raise ValueError("selection_seed must be included in model_seeds")
        if not self.candidates:
            raise ValueError("At least one TabNet candidate is required")
        if self.lambda_sparse < 0:
            raise ValueError("lambda_sparse must be non-negative")


class QATNetModel:
    def __init__(
        self,
        preprocessor: Any,
        encoder: VariationalQuantumEncoder | None,
        classifier: Any,
        mode: str,
        class_names: tuple[str, ...],
        component_count: int,
        cache_dir: Path | None = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.classifier = classifier
        self.mode = mode
        self.class_names = class_names
        self.component_count = component_count
        self.cache_dir = cache_dir

    def _features(self, records: Any, use_cache: bool = True) -> np.ndarray:
        if hasattr(records, "columns"):
            pca_values = self.preprocessor.transform(records, self.preprocessor_label_column)
        else:
            pca_values = np.asarray(records, dtype=np.float32)
        if self.mode == "amplitude":
            return amplitude_probability_features(pca_values)
        if self.mode == "qatnet":
            if self.encoder is None:
                raise RuntimeError("QATNet encoder is missing")
            if not use_cache:
                return self.encoder.transform(pca_values)
            path = self.cache_dir / "inference.npy" if self.cache_dir else None
            return _cached_quantum_features(self.encoder, pca_values, path)
        return pca_values

    def predict_proba(self, features: Any, use_cache: bool = True) -> np.ndarray:
        values = self._features(features, use_cache=use_cache)
        return np.asarray(self.classifier.predict_proba(values), dtype=np.float64)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predict_proba(features).argmax(axis=1)


def _cached_quantum_features(
    encoder: VariationalQuantumEncoder,
    values: np.ndarray,
    cache_path: Path | None,
) -> np.ndarray:
    if cache_path is None:
        return encoder.transform(values)
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(values).view(np.uint8))
    for parameter in encoder.quantum_layer.parameters():
        digest.update(parameter.detach().cpu().numpy().tobytes())
    digest.update(f"{encoder.qubits}:{encoder.layers}".encode())
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_path.with_suffix(".sha256")
    expected = digest.hexdigest()
    if cache_path.is_file() and metadata_path.is_file() and metadata_path.read_text().strip() == expected:
        return np.load(cache_path, allow_pickle=False)
    output = encoder.transform(values)
    np.save(cache_path, output, allow_pickle=False)
    metadata_path.write_text(expected + "\n")
    return output


def _balanced_indices(y: np.ndarray, limit: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels, counts = np.unique(y, return_counts=True)
    per_class = min(int(counts.min()), max(1, limit // len(labels)))
    selected = np.concatenate([
        rng.choice(np.flatnonzero(y == label), size=per_class, replace=False)
        for label in labels
    ])
    rng.shuffle(selected)
    return selected


def _new_tabnet(seed: int):
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
    except ImportError as exc:
        raise RuntimeError("Install pytorch-tabnet to train the classifier") from exc
    return TabNetClassifier(seed=seed, verbose=0)


def _paper_class_weights(y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_train)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("Training labels must be a non-empty one-dimensional array")
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return classes, weights.astype(np.float32)


def _weighted_cross_entropy(class_weights: np.ndarray, model: Any):
    device = getattr(model, "device", torch.device("cpu"))
    return torch.nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    )


def _new_paper_tabnet(seed: int, candidate: TabNetCandidate, profile: PaperTabNetProfile):
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
    except ImportError as exc:
        raise RuntimeError("Install pytorch-tabnet to run the paper TabNet profile") from exc
    return TabNetClassifier(
        n_d=candidate.n_d,
        n_a=candidate.n_a,
        n_steps=candidate.n_steps,
        mask_type=candidate.mask_type,
        lambda_sparse=profile.lambda_sparse,
        seed=seed,
        optimizer_fn=torch.optim.Adam,
        device_name=profile.device_name,
        verbose=0,
    )


def _recalibrate_batch_norm(
    network: torch.nn.Module,
    features: np.ndarray,
    batch_size: int,
) -> int:
    """Re-estimate inference BatchNorm buffers from the refit feature rows."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("BatchNorm recalibration requires non-empty 2D features")
    if batch_size < 1:
        raise ValueError("BatchNorm recalibration batch_size must be positive")
    batch_norms = [
        module for module in network.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        and module.track_running_stats
    ]
    if not batch_norms:
        return 0

    original_momenta = [module.momentum for module in batch_norms]
    device = next(network.parameters()).device
    for module in batch_norms:
        module.reset_running_stats()
        module.momentum = None

    network.train()
    try:
        with torch.no_grad():
            for start in range(0, len(values), batch_size):
                batch = torch.as_tensor(
                    values[start : start + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                network(batch)
    finally:
        for module, momentum in zip(batch_norms, original_momenta):
            module.momentum = momentum
        network.eval()
    return len(batch_norms)


def _fit_paper_candidate(
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    class_weights: np.ndarray,
    profile: PaperTabNetProfile,
) -> None:
    loss_fn = _weighted_cross_entropy(class_weights, model)
    model.fit(
        X_train=np.asarray(x_train, dtype=np.float32),
        y_train=np.asarray(y_train),
        eval_set=[(
            np.asarray(x_validation, dtype=np.float32),
            np.asarray(y_validation),
        )],
        eval_name=["validation"],
        eval_metric=["accuracy"],
        loss_fn=loss_fn,
        weights=0,
        max_epochs=profile.max_epochs,
        patience=profile.patience,
        batch_size=profile.batch_size,
        virtual_batch_size=profile.virtual_batch_size,
        num_workers=0,
        drop_last=False,
    )


def select_paper_tabnet_candidate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    class_names: Sequence[str],
    profile: PaperTabNetProfile = PaperTabNetProfile(),
    model_factory: Callable[[int, TabNetCandidate, PaperTabNetProfile], Any] | None = None,
) -> dict[str, Any]:
    """Select a candidate using train labels for loss weights and validation macro-F1 only."""
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_validation = np.asarray(x_validation)
    y_validation = np.asarray(y_validation)
    if len(x_train) != len(y_train) or len(x_validation) != len(y_validation):
        raise ValueError("Feature and label row counts must match for each split")
    if x_train.ndim != 2 or x_validation.ndim != 2 or x_train.shape[1] != x_validation.shape[1]:
        raise ValueError("Train and validation features must be 2D with matching widths")
    if not set(np.unique(y_validation)).issubset(set(np.unique(y_train))):
        raise ValueError("Validation labels must be represented in training labels")
    if len(class_names) != len(np.unique(y_train)):
        raise ValueError("class_names must match the classes present in training labels")

    labels, class_weights = _paper_class_weights(y_train)
    factory = model_factory or _new_paper_tabnet
    candidates = []
    for candidate in profile.candidates:
        model = factory(profile.selection_seed, candidate, profile)
        _fit_paper_candidate(
            model, x_train, y_train, x_validation, y_validation, class_weights, profile
        )
        probabilities = np.asarray(model.predict_proba(x_validation))
        predictions = probabilities.argmax(axis=1)
        score = float(f1_score(y_validation, predictions, average="macro", zero_division=0))
        best_epoch = max(int(getattr(model, "best_epoch", profile.max_epochs - 1)), 0)
        candidates.append({
            "candidate": candidate,
            "validation_macro_f1": score,
            "best_epoch": best_epoch,
        })

    selected = max(
        candidates,
        key=lambda result: (result["validation_macro_f1"], -result["best_epoch"]),
    )
    return {
        **selected,
        "candidate_results": candidates,
        "class_ids": labels.tolist(),
        "class_weights": class_weights.tolist(),
        "selection_seed": profile.selection_seed,
        "selection_metric": "validation_macro_f1",
    }


def run_paper_tabnet_workflow(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Sequence[str],
    split_seed: int,
    output_dir: str | Path,
    dataset_name: str,
    feature_pipeline: str = "classical",
    profile: PaperTabNetProfile = PaperTabNetProfile(),
    model_factory: Callable[[int, TabNetCandidate, PaperTabNetProfile], Any] | None = None,
) -> dict[str, Any]:
    """Select on validation, refit on train+validation, and evaluate each model seed on test.

    Input matrices must already be produced by the same training-fitted preprocessing
    and feature pipeline for all splits. `split_seed` is recorded separately from
    the model seeds; test data is consumed only after candidate selection and refits.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    selected = select_paper_tabnet_candidate(
        x_train, y_train, x_validation, y_validation, class_names,
        profile=profile, model_factory=model_factory,
    )
    candidate: TabNetCandidate = selected["candidate"]
    _, training_weights = _paper_class_weights(np.asarray(y_train))
    combined_x = np.concatenate((np.asarray(x_train), np.asarray(x_validation)), axis=0)
    combined_y = np.concatenate((np.asarray(y_train), np.asarray(y_validation)), axis=0)
    refit_epochs = selected["best_epoch"] + 1
    factory = model_factory or _new_paper_tabnet
    seed_results = []

    for model_seed in profile.model_seeds:
        model = factory(model_seed, candidate, profile)
        loss_fn = _weighted_cross_entropy(training_weights, model)
        model.fit(
            X_train=np.asarray(combined_x, dtype=np.float32),
            y_train=combined_y,
            eval_set=[],
            eval_metric=["accuracy"],
            loss_fn=loss_fn,
            weights=0,
            max_epochs=refit_epochs,
            patience=0,
            batch_size=profile.batch_size,
            virtual_batch_size=profile.virtual_batch_size,
            num_workers=0,
            drop_last=False,
        )
        recalibrated_batch_norm_layers = 0
        if feature_pipeline == "qatnet":
            recalibrated_batch_norm_layers = _recalibrate_batch_norm(
                model.network,
                combined_x,
                batch_size=profile.batch_size,
            )
        started = time.perf_counter()
        probabilities = np.asarray(model.predict_proba(np.asarray(x_test, dtype=np.float32)))
        inference_seconds_per_sample = (
            time.perf_counter() - started
        ) / max(len(y_test), 1)
        from evaluation.metrics import evaluate_predictions

        metrics = evaluate_predictions(
            np.asarray(y_test),
            probabilities.argmax(axis=1),
            probabilities,
            tuple(class_names),
        )
        result = {
            "dataset": dataset_name,
            "feature_pipeline": feature_pipeline,
            "split_seed": int(split_seed),
            "model_seed": int(model_seed),
            "selected_candidate": asdict(candidate),
            "selected_validation_macro_f1": selected["validation_macro_f1"],
            "refit_epochs": refit_epochs,
            "training_class_weights": selected["class_weights"],
            "batch_norm_recalibrated_layers": recalibrated_batch_norm_layers,
            "metrics": metrics,
            "inference_seconds_per_sample": inference_seconds_per_sample,
        }
        seed_results.append(result)
        seed_dir = output / f"seed-{model_seed}"
        seed_dir.mkdir()
        (seed_dir / "metrics.json").write_text(
            json.dumps(result, indent=2, default=float) + "\n"
        )
        joblib.dump(model, seed_dir / "classifier.joblib")

    aggregate_metrics = {}
    metric_names = (
        "accuracy", "macro_f1", "macro_auroc", "expected_calibration_error",
        "inference_seconds_per_sample",
    )
    for metric_name in metric_names:
        values = [
            item["metrics"].get(metric_name)
            if metric_name != "inference_seconds_per_sample"
            else item[metric_name]
            for item in seed_results
        ]
        values = [float(value) for value in values if value is not None]
        aggregate_metrics[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
            "n": len(values),
        } if values else {"mean": None, "std": None, "n": 0}

    report = {
        "protocol": "qatnet-paper-tabnet-profile",
        "dataset": dataset_name,
        "feature_pipeline": feature_pipeline,
        "split_seed": int(split_seed),
        "model_seeds": list(profile.model_seeds),
        "selection_seed": profile.selection_seed,
        "profile": asdict(profile),
        "selection": {
            "metric": selected["selection_metric"],
            "selected_validation_macro_f1": selected["validation_macro_f1"],
            "selected_candidate": asdict(candidate),
            "best_epoch_zero_based": selected["best_epoch"],
            "candidate_results": [
                {
                    "candidate": asdict(item["candidate"]),
                    "validation_macro_f1": item["validation_macro_f1"],
                    "best_epoch_zero_based": item["best_epoch"],
                }
                for item in selected["candidate_results"]
            ],
        },
        "refit": {
            "training_rows": len(combined_y),
            "epochs": refit_epochs,
            "class_weight_source": "original training partition only",
            "qatnet_batch_norm_calibration": (
                "Running statistics are recalibrated with unlabeled train+validation "
                "refit features after final fitting and before test inference."
            ) if feature_pipeline == "qatnet" else None,
            "assumption": (
                "The PDF says retrain on train+validation after selection but does not "
                "specify refit epoch handling. The selected candidate's best epoch from "
                "the selection run is reused as the fixed refit epoch count."
            ),
        },
        "hardware": {
            "tabnet_device_request": profile.device_name,
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "aggregate_metrics": aggregate_metrics,
        "per_seed": seed_results,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    return report


def run_paired_representation_comparison(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Sequence[str],
    split_seed: int,
    output_dir: str | Path,
    dataset_name: str,
    candidate: TabNetCandidate,
    refit_epochs: int,
    profile: PaperTabNetProfile = PaperTabNetProfile(),
    model_factory: Callable[[int, TabNetCandidate, PaperTabNetProfile], Any] | None = None,
    representation_modes: tuple[str, ...] = ("classical", "amplitude"),
    dataset_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare fixed feature representations with a shared paired TabNet setup."""
    if refit_epochs < 1:
        raise ValueError("refit_epochs must be positive")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {output}")
    labels = np.asarray(y_train)
    validation_labels = np.asarray(y_validation)
    test_labels = np.asarray(y_test)
    train_pca = np.asarray(x_train, dtype=np.float32)
    validation_pca = np.asarray(x_validation, dtype=np.float32)
    test_pca = np.asarray(x_test, dtype=np.float32)
    if any(values.ndim != 2 for values in (train_pca, validation_pca, test_pca)):
        raise ValueError("All feature partitions must be two-dimensional")
    if len(train_pca) != len(labels) or len(validation_pca) != len(validation_labels):
        raise ValueError("Training/validation feature and label row counts must match")
    if len(test_pca) != len(test_labels):
        raise ValueError("Test feature and label row counts must match")
    if not (train_pca.shape[1] == validation_pca.shape[1] == test_pca.shape[1]):
        raise ValueError("All feature partitions must have matching widths")
    if not set(np.unique(validation_labels)).issubset(set(np.unique(labels))):
        raise ValueError("Validation labels must be represented in training labels")
    if not set(np.unique(test_labels)).issubset(set(np.unique(labels))):
        raise ValueError("Test labels must be represented in training labels")
    if len(class_names) != len(np.unique(labels)):
        raise ValueError("class_names must match the classes present in training labels")

    transforms: dict[str, tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]] = {
        "classical": (
            lambda values: np.asarray(values, dtype=np.float32),
            {"formula": "identity(signed_PCA)", "normalization": "none", "sign_preserved": True},
        ),
        "normalized_signed": (
            amplitude_features,
            {
                "formula": "x / max(||x||_2, epsilon)",
                "normalization": "row-wise L2",
                "epsilon": 1e-12,
                "zero_row_fallback": "first coordinate set to 1; remaining coordinates 0",
                "sign_preserved": True,
            },
        ),
        "amplitude": (
            amplitude_probability_features,
            {
                "formula": "square(x / max(||x||_2, epsilon))",
                "normalization": "row-wise L2 followed by elementwise square",
                "epsilon": 1e-12,
                "zero_row_fallback": "first coordinate probability 1; remaining coordinates 0",
                "sign_preserved": False,
            },
        ),
    }
    if not representation_modes or len(set(representation_modes)) != len(representation_modes):
        raise ValueError("representation_modes must be a non-empty sequence of unique names")
    unknown_modes = set(representation_modes) - set(transforms)
    if unknown_modes:
        raise ValueError(f"Unknown representation mode(s): {sorted(unknown_modes)}")
    representations = {}
    for mode in representation_modes:
        transform, _ = transforms[mode]
        representations[mode] = tuple(
            np.asarray(transform(values), dtype=np.float32)
            for values in (train_pca, validation_pca, test_pca)
        )
    class_ids, class_weights = _paper_class_weights(labels)
    factory = model_factory or _new_paper_tabnet
    combined_labels = np.concatenate((labels, validation_labels))
    rows_by_mode: dict[str, list[dict[str, Any]]] = {}
    output.mkdir(parents=True, exist_ok=False)

    for mode, (mode_train, mode_validation, mode_test) in representations.items():
        mode_dir = output / mode
        mode_dir.mkdir()
        transform_config = {
            "representation": mode,
            "input": "16-dimensional signed PCA coordinates, fitted using training partition only",
            **transforms[mode][1],
        }
        (mode_dir / "transformation.json").write_text(
            json.dumps(transform_config, indent=2) + "\n"
        )
        if dataset_manifest is not None:
            (mode_dir / "dataset-manifest.json").write_text(
                json.dumps(dataset_manifest, indent=2) + "\n"
            )
        results = []
        prediction_rows = []
        for seed in profile.model_seeds:
            seed_dir = mode_dir / f"seed-{seed}"
            seed_dir.mkdir()
            selection_model = factory(seed, candidate, profile)
            selection_model.fit(
                X_train=mode_train,
                y_train=labels,
                eval_set=[(mode_validation, validation_labels)],
                eval_name=["validation"],
                eval_metric=["logloss", "accuracy"],
                loss_fn=_weighted_cross_entropy(class_weights, selection_model),
                weights=0,
                max_epochs=profile.max_epochs,
                patience=profile.patience,
                batch_size=profile.batch_size,
                virtual_batch_size=profile.virtual_batch_size,
                num_workers=0,
                drop_last=False,
            )
            best_epoch = max(int(getattr(selection_model, "best_epoch", 0)), 0)
            history = selection_model.history.history
            curves = {
                "epoch": list(range(len(history["loss"]))),
                "training_loss": [float(value) for value in history["loss"]],
                "validation_logloss": [
                    float(value) for value in history["validation_logloss"]
                ],
                "validation_accuracy": [
                    float(value) for value in history["validation_accuracy"]
                ],
            }
            pd.DataFrame(curves).to_csv(seed_dir / "selection-learning-curves.csv", index=False)
            joblib.dump(selection_model, seed_dir / "selection-classifier.joblib")

            training_probabilities = np.asarray(selection_model.predict_proba(mode_train))
            validation_probabilities = np.asarray(
                selection_model.predict_proba(mode_validation)
            )
            from evaluation.metrics import evaluate_predictions

            training_metrics = evaluate_predictions(
                labels,
                training_probabilities.argmax(axis=1),
                training_probabilities,
                tuple(class_names),
            )
            validation_metrics = evaluate_predictions(
                validation_labels,
                validation_probabilities.argmax(axis=1),
                validation_probabilities,
                tuple(class_names),
            )
            training_predictions = training_probabilities.argmax(axis=1)
            validation_predictions = validation_probabilities.argmax(axis=1)

            final_model = factory(seed, candidate, profile)
            combined_features = np.concatenate((mode_train, mode_validation), axis=0)
            final_model.fit(
                X_train=combined_features,
                y_train=combined_labels,
                eval_set=[],
                eval_metric=["accuracy"],
                loss_fn=_weighted_cross_entropy(class_weights, final_model),
                weights=0,
                max_epochs=refit_epochs,
                patience=0,
                batch_size=profile.batch_size,
                virtual_batch_size=profile.virtual_batch_size,
                num_workers=0,
                drop_last=False,
            )
            test_probabilities = np.asarray(final_model.predict_proba(mode_test))
            test_metrics = evaluate_predictions(
                test_labels,
                test_probabilities.argmax(axis=1),
                test_probabilities,
                tuple(class_names),
            )
            joblib.dump(final_model, seed_dir / "classifier.joblib")

            result = {
                "dataset": dataset_name,
                "representation": mode,
                "split_seed": int(split_seed),
                "model_seed": int(seed),
                "candidate": asdict(candidate),
                "class_weights_from_training_only": class_weights.tolist(),
                "selection_best_epoch_zero_based": best_epoch,
                "fixed_refit_epochs": refit_epochs,
                "transformation": transform_config,
                "training_metrics_at_validation_selected_checkpoint": training_metrics,
                "validation_metrics_at_validation_selected_checkpoint": validation_metrics,
                "test_metrics_after_fixed_train_plus_validation_refit": test_metrics,
                "predicted_class_counts": {
                    "training_at_validation_selected_checkpoint": {
                        name: int(np.count_nonzero(training_predictions == index))
                        for index, name in enumerate(class_names)
                    },
                    "validation_at_validation_selected_checkpoint": {
                        name: int(np.count_nonzero(validation_predictions == index))
                        for index, name in enumerate(class_names)
                    },
                    "test_after_fixed_refit": {
                        name: int(np.count_nonzero(test_probabilities.argmax(axis=1) == index))
                        for index, name in enumerate(class_names)
                    },
                },
            }
            (seed_dir / "metrics.json").write_text(
                json.dumps(result, indent=2, default=float) + "\n"
            )
            (seed_dir / "run-manifest.json").write_text(json.dumps({
                "dataset": dataset_name,
                "representation": mode,
                "transformation": transform_config,
                "split_seed": int(split_seed),
                "model_seed": int(seed),
                "candidate": asdict(candidate),
                "training_config": {
                    "optimizer": "Adam",
                    "loss": "class-weighted CrossEntropyLoss",
                    "class_weights_from_training_only": class_weights.tolist(),
                    "max_epochs_selection": profile.max_epochs,
                    "patience": profile.patience,
                    "batch_size": profile.batch_size,
                    "virtual_batch_size": profile.virtual_batch_size,
                    "selection_metric": "validation_accuracy",
                    "refit_epochs": refit_epochs,
                    "refit_data": "train + validation",
                },
                "dataset_manifest": dataset_manifest,
            }, indent=2, default=float) + "\n")
            if dataset_manifest is not None:
                (seed_dir / "dataset-manifest.json").write_text(
                    json.dumps(dataset_manifest, indent=2) + "\n"
                )
            prediction_rows.extend({
                "representation": mode,
                "model_seed": seed,
                "test_row_index": row,
                "true_class": class_names[int(test_labels[row])],
                "predicted_class": class_names[int(test_probabilities[row].argmax())],
                **{
                    f"probability_{name}": float(test_probabilities[row, class_index])
                    for class_index, name in enumerate(class_names)
                },
            } for row in range(len(test_labels)))
            results.append(result)
        pd.DataFrame(prediction_rows).to_csv(mode_dir / "test-predictions.csv", index=False)
        (mode_dir / "predicted-class-distributions.json").write_text(json.dumps({
            str(result["model_seed"]): result["predicted_class_counts"]
            for result in results
        }, indent=2) + "\n")
        rows_by_mode[mode] = results

    metric_fields = (
        "accuracy", "macro_precision", "macro_recall", "macro_f1",
        "macro_auroc", "expected_calibration_error"
    )
    aggregates = {}
    for mode, results in rows_by_mode.items():
        aggregates[mode] = {}
        for partition_key in (
            "training_metrics_at_validation_selected_checkpoint",
            "validation_metrics_at_validation_selected_checkpoint",
            "test_metrics_after_fixed_train_plus_validation_refit",
        ):
            aggregates[mode][partition_key] = {}
            for metric in metric_fields:
                values = [
                    (
                        result[partition_key].get("per_class", {})
                        .get("macro avg", {})
                        .get(metric.removeprefix("macro_"))
                    )
                    if metric in {"macro_precision", "macro_recall"}
                    else result[partition_key].get(metric)
                    for result in results
                ]
                values = [float(value) for value in values if value is not None]
                aggregates[mode][partition_key][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=0)),
                    "n": len(values),
                } if values else {"mean": None, "std": None, "n": 0}

    report = {
        "protocol": "fixed-tabnet-representation-ablation-pilot",
        "dataset": dataset_name,
        "split_seed": int(split_seed),
        "model_seeds": list(profile.model_seeds),
        "class_names": list(class_names),
        "representations": {
            mode: {
                "transformation": transforms[mode][1],
                "feature_count": int(representations[mode][0].shape[1]),
            }
            for mode in representation_modes
        },
        "dataset_manifest": dataset_manifest,
        "candidate_fixed_from_existing_classical_baseline": asdict(candidate),
        "selection_policy": (
            "No architecture sweep; each seed fits the same fixed architecture on train, "
            "uses validation accuracy for early stopping, and records validation logloss."
        ),
        "refit_policy": (
            "Each representation and seed refits the same architecture for the same fixed "
            "epoch count on train+validation; 57 epochs carried forward from the existing "
            "classical baseline to avoid representation-specific refit durations."
        ),
        "training_class_ids": class_ids.tolist(),
        "training_class_weights": class_weights.tolist(),
        "training_config": {
            "optimizer": "Adam",
            "max_epochs_selection": profile.max_epochs,
            "patience": profile.patience,
            "batch_size": profile.batch_size,
            "virtual_batch_size": profile.virtual_batch_size,
            "selection_early_stopping_metric": "validation_accuracy",
            "validation_loss_metric": "logloss",
            "fixed_refit_epochs": refit_epochs,
            "refit_uses_validation_for_training": True,
            "test_use": "final scoring only",
        },
        "per_representation_per_seed": rows_by_mode,
        "aggregates": aggregates,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    return report


def train_pipeline(
    splits: DataSplits,
    mode: str,
    config: TrainingConfig,
    cache_dir: str | Path | None = None,
) -> tuple[QATNetModel, dict[str, Any]]:
    if mode not in {"classical", "amplitude", "qatnet"}:
        raise ValueError(f"Unknown pipeline mode: {mode}")
    if mode == "qatnet" and config.pretrain_samples < len(splits.class_names):
        raise ValueError("pretrain_samples must be at least the number of classes")
    if mode != "classical" and splits.x_train.shape[1] != 2**config.qubits:
        raise ValueError(f"Pipeline Q={config.qubits} requires {2**config.qubits} PCA components")
    timings: dict[str, float] = {"vqc_pretraining_seconds": 0.0, "feature_generation_seconds": 0.0}
    encoder = None
    if mode == "classical":
        train_x, val_x, test_x = splits.x_train, splits.x_validation, splits.x_test
    elif mode == "amplitude":
        started = time.perf_counter()
        train_x, val_x, test_x = map(
            amplitude_probability_features,
            (splits.x_train, splits.x_validation, splits.x_test),
        )
        timings["feature_generation_seconds"] = time.perf_counter() - started
    else:
        encoder = VariationalQuantumEncoder(config.qubits, config.vqc_layers, config.seed)
        selected = _balanced_indices(splits.y_train, config.pretrain_samples, config.seed)
        started = time.perf_counter()
        losses = encoder.pretrain(
            splits.x_train[selected], splits.y_train[selected],
            epochs=config.pretrain_epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            seed=config.seed,
        )
        timings["vqc_pretraining_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        cache_root = Path(cache_dir) if cache_dir is not None else None
        train_x, val_x, test_x = (
            _cached_quantum_features(encoder, splits.x_train, cache_root / "train.npy" if cache_root else None),
            _cached_quantum_features(encoder, splits.x_validation, cache_root / "validation.npy" if cache_root else None),
            _cached_quantum_features(encoder, splits.x_test, cache_root / "test.npy" if cache_root else None),
        )
        timings["feature_generation_seconds"] = time.perf_counter() - started
        timings["vqc_pretraining_losses"] = losses

    model = _new_tabnet(config.seed)
    started = time.perf_counter()
    model.fit(
        X_train=np.asarray(train_x, dtype=np.float32),
        y_train=splits.y_train,
        eval_set=[(np.asarray(val_x, dtype=np.float32), splits.y_validation)],
        eval_name=["validation"],
        eval_metric=["accuracy"],
        max_epochs=config.tabnet_max_epochs,
        patience=min(config.tabnet_patience, config.tabnet_max_epochs),
        batch_size=max(8, min(config.batch_size, len(train_x))),
        virtual_batch_size=max(8, min(128, config.batch_size, len(train_x))),
        num_workers=0,
        drop_last=False,
    )
    timings["classifier_training_seconds"] = time.perf_counter() - started
    timings["training_time_seconds"] = sum(
        timings[key] for key in (
            "vqc_pretraining_seconds", "feature_generation_seconds", "classifier_training_seconds"
        )
    )
    result = QATNetModel(
        splits.preprocessor, encoder, model, mode, splits.class_names,
        splits.preprocessor.component_count,
        Path(cache_dir) if cache_dir is not None and encoder is not None else None,
    )
    result.preprocessor_label_column = splits.report.label_column
    return result, timings


def save_model(model: QATNetModel, config: TrainingConfig, output_dir: str | Path, timings: dict[str, Any]) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model.preprocessor, output / "preprocessor.joblib")
    joblib.dump(model.classifier, output / "classifier.joblib")
    if model.encoder is not None:
        model.encoder.torch.save(
            model.encoder.quantum_layer.state_dict(),
            output / "quantum_encoder.pt",
        )
    package_versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "torch", "pytorch-tabnet", "pennylane", "qiskit", "qiskit-aer"):
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = "not-installed"
    metadata = {
        "mode": model.mode,
        "qubits": config.qubits,
        "vqc_layers": config.vqc_layers,
        "class_names": list(model.class_names),
        "component_count": model.component_count,
        "label_column": getattr(model, "preprocessor_label_column", "Label"),
        "feature_manifest": list(getattr(model.preprocessor, "feature_names", ())),
        "input_feature_manifest": list(getattr(model.preprocessor, "input_feature_names", ())),
        "random_seed": config.seed,
        "training_config": asdict(config),
        "training_metadata": timings,
        "software": {"python": __import__("sys").version.split()[0], "packages": package_versions},
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
            "tabnet_device": "cuda" if torch.cuda.is_available() else "cpu",
            "quantum_simulator": "PennyLane default.qubit (CPU)" if model.encoder else None,
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=float) + "\n")
    return output


def load_model(model_dir: str | Path) -> QATNetModel:
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "metadata.json").read_text())
    preprocessor = joblib.load(model_dir / "preprocessor.joblib")
    classifier = joblib.load(model_dir / "classifier.joblib")
    encoder = None
    if metadata["mode"] == "qatnet":
        encoder = VariationalQuantumEncoder(
            int(metadata["qubits"]),
            int(metadata["vqc_layers"]),
            int(metadata["random_seed"]),
        )
        weights = encoder.torch.load(
            model_dir / "quantum_encoder.pt", map_location="cpu", weights_only=True
        )
        encoder.quantum_layer.load_state_dict(weights)
        for parameter in encoder.quantum_layer.parameters():
            parameter.requires_grad_(False)
        encoder.quantum_layer.eval()
    model = QATNetModel(
        preprocessor, encoder, classifier, metadata["mode"],
        tuple(metadata["class_names"]), int(metadata["component_count"]),
        model_dir / "feature_cache" if encoder is not None else None,
    )
    model.preprocessor_label_column = metadata["label_column"]
    return model
