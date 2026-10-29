from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import torch

from data.dataset import (
    PAPER_CLASS_MAPPING,
    PAPER_CLASSES,
    inspect_dataset,
    load_csv,
    prepare_paper_dataset,
    prepare_splits,
    sample_csv_files,
)
from evaluation.metrics import evaluate_predictions
from evaluation.plots import save_confusion_matrix, save_metric_plot
from noise.experiment import qiskit_noisy_amplitude_features, train_noisy_feature_learners
from qatnet.training import TrainingConfig, save_model, train_pipeline
from qatnet.training import (
    PaperTabNetProfile,
    TabNetCandidate,
    run_paired_representation_comparison,
    run_paper_tabnet_workflow,
)
from qatnet.circuit import VariationalQuantumEncoder, amplitude_probability_features
from qatnet.service import QATNetDetectionService, telemetry_from_cic_records
from qkd.simulator import QKDConfig, QKDSimulator
from quasar.demo import run_demo
from quasar.qatnet_demo import run_qatnet_replay

logger = logging.getLogger(__name__)


def train(args: argparse.Namespace) -> None:
    frame = load_csv(args.dataset, args.label)
    if args.sample_rows:
        print(f"Development sampling enabled: at most {args.sample_rows} stratified rows.")
    root = Path(args.output)
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    (root / "plots").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(parents=True, exist_ok=True)
    all_results = []
    for qubits in (4, 5, 6):
        splits = prepare_splits(
            frame, args.label, components=2**qubits,
            seed=args.seed, sample_rows=args.sample_rows,
        )
        for mode in ("classical", "amplitude", "qatnet"):
            config = TrainingConfig(
                qubits=qubits,
                pretrain_samples=args.pretrain_samples,
                pretrain_epochs=args.pretrain_epochs,
                batch_size=args.batch_size,
                tabnet_max_epochs=args.epochs,
                tabnet_patience=min(15, args.epochs),
                seed=args.seed,
            )
            cache_dir = root / "cache" / f"qatnet-q{qubits}" if mode == "qatnet" else None
            model, timings = train_pipeline(splits, mode, config, cache_dir=cache_dir)
            inference_started = time.perf_counter()
            probabilities = model.predict_proba(splits.raw_x_test, use_cache=False)
            inference_latency = (
                time.perf_counter() - inference_started
            ) / max(len(splits.raw_x_test), 1)
            predictions = probabilities.argmax(axis=1)
            metrics = evaluate_predictions(splits.y_test, predictions, probabilities, splits.class_names)
            result = {
                "model": mode,
                "qubits": qubits,
                **metrics,
                **timings,
                "inference_latency_seconds_per_sample": inference_latency,
                "dataset_rows": splits.report.rows,
                "split_sizes": {
                    "train": len(splits.y_train),
                    "validation": len(splits.y_validation),
                    "test": len(splits.y_test),
                },
                "class_names": list(splits.class_names),
                "seed": args.seed,
                "data_kind": args.data_kind,
                "hardware": {
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                    "processor": platform.processor(),
                    "torch_cuda_available": bool(torch.cuda.is_available()),
                    "torch_mps_available": bool(
                        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                    ),
                    "tabnet_device": "cuda" if torch.cuda.is_available() else "cpu",
                    "quantum_device": "PennyLane default.qubit (CPU)",
                    "noise_simulator": "Qiskit Aer CPU",
                },
                "software": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "scikit_learn": __import__("sklearn").__version__,
                    "torch": __import__("torch").__version__,
                    "pennylane": __import__("pennylane").__version__,
                    "qiskit": __import__("qiskit").__version__,
                },
            }
            all_results.append(result)
            model_dir = root / "models" / f"{mode}-q{qubits}"
            save_model(model, config, model_dir, timings)
            (root / "metrics" / f"feature-manifest-q{qubits}.json").write_text(json.dumps({
                "dataset": str(Path(args.dataset)),
                "label_column": args.label,
                "class_mapping": {name: index for index, name in enumerate(splits.class_names)},
                "feature_names": list(splits.feature_names),
                "dropped_identifiers": list(splits.report.dropped_identifiers),
                "pca_components_requested": 2**qubits,
                "pca_components_fitted": splits.preprocessor.component_count,
                "seed": args.seed,
            }, indent=2) + "\n")
            pd.DataFrame({
                "y_true": splits.y_test,
                "y_pred": predictions,
                **{f"probability_{name}": probabilities[:, index] for index, name in enumerate(splits.class_names)},
            }).to_csv(root / "metrics" / f"predictions-{mode}-q{qubits}.csv", index=False)
            save_confusion_matrix(
                metrics["confusion_matrix"], splits.class_names,
                root / "plots" / f"confusion-{mode}-q{qubits}.png",
            )
            logger.info(
                "%s Q=%d: test accuracy=%.4f, macro-F1=%.4f",
                mode, qubits, metrics["accuracy"], metrics["macro_f1"],
            )
    flat_results = []
    for result in all_results:
        flat_results.append({
            "model": result["model"],
            "qubits": result["qubits"],
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
            "macro_auroc": result["macro_auroc"],
            "vqc_pretraining_seconds": result["vqc_pretraining_seconds"],
            "feature_generation_seconds": result["feature_generation_seconds"],
            "classifier_training_seconds": result["classifier_training_seconds"],
            "training_time_seconds": result["training_time_seconds"],
            "inference_latency_seconds_per_sample": result["inference_latency_seconds_per_sample"],
            "expected_calibration_error": result["expected_calibration_error"],
            "per_class_json": json.dumps(result["per_class"], sort_keys=True),
            "confusion_matrix_json": json.dumps(result["confusion_matrix"]),
            "seed": result["seed"],
        })
    pd.DataFrame(flat_results).to_csv(root / "metrics" / "results.csv", index=False)
    (root / "metrics" / "results.json").write_text(json.dumps(all_results, indent=2, default=float) + "\n")
    (root / "metrics" / "dataset-report.json").write_text(
        json.dumps(asdict(inspect_dataset(frame, args.label)), indent=2) + "\n"
    )
    save_metric_plot(flat_results, root / "plots" / "macro-f1.png")


def model_predict_latency(model, values: np.ndarray) -> float:
    started = time.perf_counter()
    model.predict_proba(values)
    return (time.perf_counter() - started) / max(len(values), 1)


def inspect(args: argparse.Namespace) -> None:
    frame = load_csv(args.dataset, args.label)
    report = asdict(inspect_dataset(frame, args.label))
    logger.info("Dataset report:\n%s", json.dumps(report, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")


def paper_smoke(args: argparse.Namespace) -> None:
    source_files = sorted(Path(args.data_dir).glob("*.csv"))
    if not source_files:
        raise FileNotFoundError(f"No CSV files found under {args.data_dir}")
    output = Path(args.output or (
        Path("outputs/runs") / f"paper-smoke-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    ))
    output.mkdir(parents=True, exist_ok=False)
    file_handler = logging.FileHandler(output / "run.log")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    ))
    logger.addHandler(file_handler)

    try:
        logger.info(
            "Streaming %d CSVs for %d samples per class (seed=%d)",
            len(source_files), args.samples_per_class, args.seed,
        )
        sampled = sample_csv_files(
            source_files,
            samples_per_class=args.samples_per_class,
            seed=args.seed,
            label_column=args.label,
            label_mapping=PAPER_CLASS_MAPPING,
            target_classes=PAPER_CLASSES,
            chunk_size=args.chunk_size,
        )
        sampled.frame.to_csv(output / "sampled.csv", index=False)
        logger.info(
            "Stream complete: source rows=%d, sampled counts=%s, repeated headers=%d",
            sum(sampled.report.source_row_counts.values()),
            sampled.report.sampled_label_counts,
            sampled.report.repeated_header_rows,
        )
        splits = prepare_splits(
            sampled.frame,
            label_column=args.label,
            components=16,
            seed=args.split_seed,
        )
        config = TrainingConfig(
            qubits=4,
            tabnet_max_epochs=args.epochs,
            tabnet_patience=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        model, timings = train_pipeline(splits, "classical", config)
        probabilities = model.predict_proba(splits.x_test)
        metrics = evaluate_predictions(
            splits.y_test, probabilities.argmax(axis=1), probabilities, splits.class_names
        )
        full_balance = args.samples_per_class == 40_000
        metadata = {
            "experiment": (
                "five-class 200,000-row PCA-TabNet baseline"
                if full_balance else
                "paper-protocol PCA-TabNet smoke test; not final results"
            ),
            "paper_reference": "https://tqe.ieee.org/2026/03/24/qatnet-a-lightweight-quantum-classical-tabular-network-for-low-latency-intrusion-detection/",
            "paper_class_mapping": PAPER_CLASS_MAPPING,
            "source_files": list(sampled.report.source_files),
            "source_row_counts": sampled.report.source_row_counts,
            "original_label_counts": sampled.report.original_label_counts,
            "normalized_label_counts": sampled.report.normalized_label_counts,
            "sampled_label_counts": sampled.report.sampled_label_counts,
            "split_class_counts": splits.split_class_counts,
            "repeated_header_rows": sampled.report.repeated_header_rows,
            "missing_values": sampled.report.missing_values,
            "infinite_values": sampled.report.infinite_values,
            "excluded_columns": list(sampled.report.excluded_columns),
            "feature_names": list(splits.feature_names),
            "sampling_seed": args.seed,
            "split_seed": args.split_seed,
            "samples_per_class": args.samples_per_class,
            "paper_balance_size_reached": full_balance,
            "duplicate_policy": sampled.report.duplicate_policy,
            "preprocessing": {
                "fit_partition": "train only",
                "imputation": "numeric median; categorical most frequent",
                "scaling": "StandardScaler on numeric training features",
                "pca_components_requested": 16,
                "pca_components_fitted": splits.preprocessor.component_count,
            },
            "training_config": asdict(config),
            "training_timings": timings,
            "smoke_test_metrics": metrics,
            "interpretation": (
                "Full-size baseline output; exact reproduction claim remains pending protocol verification."
                if full_balance else
                "Development smoke output only; do not cite as paper reproduction results."
            ),
        }
        (output / "run.json").write_text(json.dumps(metadata, indent=2, default=float) + "\n")
        (output / "sampled-label-counts.json").write_text(
            json.dumps(sampled.report.sampled_label_counts, indent=2) + "\n"
        )
        save_model(model, config, output / "model", timings)
        logger.info("Smoke test complete; run artifacts written to %s", output)
    finally:
        logger.removeHandler(file_handler)
        file_handler.close()


def explore_sample(args: argparse.Namespace) -> None:
    source_files = sorted(Path(args.data_dir).glob("*.csv"))
    if not source_files:
        raise FileNotFoundError(f"No CSV files found under {args.data_dir}")
    output = Path(args.output or (
        Path("outputs/runs") / f"all-label-sample-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    ))
    output.mkdir(parents=True, exist_ok=False)
    sample = sample_csv_files(
        source_files,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
        label_column=args.label,
        chunk_size=args.chunk_size,
        require_samples_per_class=False,
    )
    sample.frame.to_csv(output / "sampled-all-labels.csv", index=False)
    (output / "dataset-report.json").write_text(json.dumps(asdict(sample.report), indent=2) + "\n")
    logger.info(
        "Saved all-label exploratory sample to %s; counts=%s",
        output, sample.report.sampled_label_counts,
    )


def prepare_cic_pilot(args: argparse.Namespace) -> None:
    source_files = [Path(args.data_dir) / name for name in args.source_files]
    manifest = prepare_paper_dataset(
        source_files,
        output_dir=args.output,
        samples_per_class=args.samples_per_class,
        sampling_seed=args.seed,
        split_seed=args.split_seed,
        label_column=args.label,
        chunk_size=args.chunk_size,
    )
    logger.info(
        "Prepared CIC pilot: %s; rows=%d; classes=%s; split_counts=%s",
        args.output,
        sum(manifest["sampling"]["sampled_class_counts"].values()),
        manifest["sampling"]["sampled_class_counts"],
        manifest["split"]["class_counts"],
    )


def paper_tabnet(args: argparse.Namespace) -> None:
    source = Path(args.dataset)
    if not source.is_file():
        raise FileNotFoundError(f"Balanced sample CSV not found: {source}")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {output}")
    frame = load_csv(source, args.label)
    splits = prepare_splits(
        frame,
        label_column=args.label,
        components=2**args.qubits,
        seed=args.split_seed,
    )
    split_features = (splits.x_train, splits.x_validation, splits.x_test)
    if args.mode == "amplitude":
        split_features = tuple(map(amplitude_probability_features, split_features))
    elif args.mode == "qatnet":
        encoder = VariationalQuantumEncoder(args.qubits, layers=1, seed=101)
        encoder.pretrain(
            splits.x_train,
            splits.y_train,
            epochs=15,
            batch_size=256,
            seed=101,
        )
        split_features = tuple(encoder.transform(values) for values in split_features)

    report = run_paper_tabnet_workflow(
        split_features[0],
        splits.y_train,
        split_features[1],
        splits.y_validation,
        split_features[2],
        splits.y_test,
        splits.class_names,
        split_seed=args.split_seed,
        output_dir=output,
        dataset_name=args.dataset_name,
        feature_pipeline=args.mode,
        profile=PaperTabNetProfile(),
    )
    (output / "data-split.json").write_text(json.dumps({
        "dataset": args.dataset_name,
        "source_csv": str(source),
        "split_seed": args.split_seed,
        "class_names": list(splits.class_names),
        "split_class_counts": splits.split_class_counts,
        "feature_names": list(splits.feature_names),
        "excluded_identifiers": list(splits.report.dropped_identifiers),
        "pca_components_requested": 2**args.qubits,
        "pca_components_fitted": splits.preprocessor.component_count,
        "preprocessing_fit_partition": "train only",
    }, indent=2) + "\n")
    logger.info(
        "Paper-profile TabNet workflow complete for %s / %s; summary: %s",
        report["dataset"], report["feature_pipeline"], output / "summary.json",
    )


def paired_tabnet(args: argparse.Namespace) -> None:
    source = Path(args.dataset)
    if not source.is_file():
        raise FileNotFoundError(f"Balanced sample CSV not found: {source}")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {output}")
    frame = load_csv(source, args.label)
    splits = prepare_splits(
        frame,
        label_column=args.label,
        components=2**args.qubits,
        seed=args.split_seed,
    )
    profile = PaperTabNetProfile()
    candidate = TabNetCandidate(n_d=8, n_a=8, n_steps=3, mask_type="sparsemax")
    split_manifest = {
        "dataset": args.dataset_name,
        "source_csv": str(source),
        "source_csv_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "split_seed": args.split_seed,
        "class_names": list(splits.class_names),
        "split_class_counts": splits.split_class_counts,
        "feature_names": list(splits.feature_names),
        "excluded_identifiers": list(splits.report.dropped_identifiers),
        "pca_components_requested": 2**args.qubits,
        "pca_components_fitted": splits.preprocessor.component_count,
        "preprocessing_fit_partition": "train only",
        "train_feature_label_sha256": _partition_fingerprint(splits.x_train, splits.y_train),
        "validation_feature_label_sha256": _partition_fingerprint(
            splits.x_validation, splits.y_validation
        ),
        "test_feature_label_sha256": _partition_fingerprint(splits.x_test, splits.y_test),
        "rows": {
            "train": len(splits.y_train),
            "validation": len(splits.y_validation),
            "test": len(splits.y_test),
        },
    }
    report = run_paired_representation_comparison(
        splits.x_train,
        splits.y_train,
        splits.x_validation,
        splits.y_validation,
        splits.x_test,
        splits.y_test,
        splits.class_names,
        split_seed=args.split_seed,
        output_dir=output,
        dataset_name=args.dataset_name,
        candidate=candidate,
        refit_epochs=57,
        profile=profile,
        representation_modes=getattr(
            args, "representation_modes", ("classical", "amplitude")
        ),
        dataset_manifest=split_manifest,
    )
    (output / "data-split.json").write_text(json.dumps(split_manifest, indent=2) + "\n")
    logger.info(
        "Paired fixed-TabNet comparison complete; summary: %s",
        output / "summary.json",
    )


def _partition_fingerprint(features: np.ndarray, labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(features).view(np.uint8))
    digest.update(np.ascontiguousarray(labels).view(np.uint8))
    return digest.hexdigest()


def qatnet_quasar_smoke(args: argparse.Namespace) -> None:
    sample_path = Path(args.sample_csv)
    if not sample_path.is_file():
        raise FileNotFoundError(f"Sample CSV not found: {sample_path}")
    output = Path(args.output or (
        Path("outputs/runs") / f"quasar-qatnet-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    ))
    output.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(sample_path, low_memory=False)
    splits = prepare_splits(
        frame,
        label_column=args.label,
        components=16,
        seed=args.split_seed,
    )
    config = TrainingConfig(
        qubits=4,
        vqc_layers=1,
        pretrain_samples=args.pretrain_samples,
        pretrain_epochs=args.pretrain_epochs,
        batch_size=args.batch_size,
        tabnet_max_epochs=args.epochs,
        tabnet_patience=args.epochs,
        seed=args.seed,
    )
    model, timings = train_pipeline(
        splits,
        "qatnet",
        config,
        cache_dir=output / "feature-cache",
    )
    model_dir = save_model(model, config, output / "model", timings)
    service = QATNetDetectionService.from_model_dir(
        model_dir,
        malicious_labels=set(splits.class_names) - {"Benign"},
    )
    count = min(args.replay_flows, len(splits.raw_x_test))
    records = splits.raw_x_test.iloc[:count].copy()
    flows = telemetry_from_cic_records(records, id_prefix="cic-test-replay")
    result = run_qatnet_replay(
        service,
        flows,
        records,
        output / "quasar",
        seed=args.seed,
    )
    report = {
        "experiment": "QATNet + QUASAR simulated integration smoke; not final paper results",
        "sample_csv": str(sample_path),
        "class_names": list(splits.class_names),
        "split_class_counts": splits.split_class_counts,
        "pca_components_fitted": splits.preprocessor.component_count,
        "training_config": asdict(config),
        "training_timings": timings,
        "model_version": service.model_version,
        "replay_flow_count": count,
        "risk_score": result["assessment"]["risk_score"],
        "mitigation_status": result["mitigation"]["status"],
        "live_network_actions": False,
    }
    (output / "integration-smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.info(
        "QATNet/QUASAR offline replay complete: %d flows, risk=%.3f, result=%s",
        count, result["assessment"]["risk_score"], output,
    )


def noise_experiment(args: argparse.Namespace) -> None:
    splits = prepare_splits(
        load_csv(args.dataset, args.label),
        label_column=args.label,
        components=2**args.qubits,
        seed=args.seed,
        sample_rows=args.sample_rows,
    )
    started = time.perf_counter()
    train_features = qiskit_noisy_amplitude_features(
        splits.x_train, args.shots, args.depolarizing_probability, args.seed
    )
    test_features = qiskit_noisy_amplitude_features(
        splits.x_test, args.shots, args.depolarizing_probability, args.seed + 1
    )
    feature_time = time.perf_counter() - started
    outputs = train_noisy_feature_learners(
        train_features, splits.y_train, test_features, seed=args.seed
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for learner, probabilities in outputs.items():
        predictions = probabilities.argmax(axis=1)
        result = evaluate_predictions(splits.y_test, predictions, probabilities, splits.class_names)
        result["feature_generation_seconds"] = feature_time
        results[learner] = result
        pd.DataFrame({
            "y_true": splits.y_test,
            "y_pred": predictions,
            **{
                f"probability_{name}": probabilities[:, index]
                for index, name in enumerate(splits.class_names)
            },
        }).to_csv(output / f"predictions-{learner}.csv", index=False)
    (output / "noisy-feature-results.json").write_text(json.dumps({
        "experiment": "separate noisy amplitude-feature learners; not noisy full QATNet",
        "data_kind": args.data_kind,
        "qubits": args.qubits,
        "shots": args.shots,
        "depolarizing_probability": args.depolarizing_probability,
        "seed": args.seed,
        "results": results,
    }, indent=2, default=float) + "\n")
    pd.DataFrame([{
        "learner": name,
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
        "macro_auroc": result["macro_auroc"],
        "expected_calibration_error": result["expected_calibration_error"],
        "feature_generation_seconds": feature_time,
    } for name, result in results.items()]).to_csv(output / "noisy-feature-results.csv", index=False)
    logger.info("Noisy-feature experiment results written to %s", output)


def simulate_qkd(args: argparse.Namespace) -> None:
    simulator = QKDSimulator(QKDConfig(seed=args.seed, initial_key_pool_bits=args.initial_pool))
    simulator.run(args.steps, args.scenario)
    simulator.export(args.csv, args.json)
    logger.info("Wrote %d synthetic QKD samples to %s", len(simulator.records), args.csv)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="qatnet-quasar")
    subparsers = parser.add_subparsers(dest="command", required=True)
    training = subparsers.add_parser("train", help="Train/evaluate all three pipelines for Q=4,5,6")
    training.add_argument("--dataset", required=True, help="Path to one CIC-IDS2018-style CSV export")
    training.add_argument("--label", default="Label")
    training.add_argument("--sample-rows", type=int)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--data-kind", choices=("real", "synthetic", "unspecified"), default="unspecified")
    training.add_argument("--pretrain-samples", type=int, default=3000)
    training.add_argument("--pretrain-epochs", type=int, default=15)
    training.add_argument("--batch-size", type=int, default=256)
    training.add_argument("--epochs", type=int, default=100)
    training.add_argument("--output", default="outputs")
    training.set_defaults(function=train)
    inspection = subparsers.add_parser("inspect", help="Inspect labels, class counts, missing/infinite values")
    inspection.add_argument("--dataset", required=True)
    inspection.add_argument("--label", default="Label")
    inspection.add_argument("--output", default="outputs/metrics/dataset-report.json")
    inspection.set_defaults(function=inspect)
    smoke = subparsers.add_parser(
        "paper-smoke",
        help="Stream CIC-IDS2018 and run a small five-class Q=4 PCA-TabNet smoke test",
    )
    smoke.add_argument("--data-dir", default="cc")
    smoke.add_argument("--label", default="Label")
    smoke.add_argument("--samples-per-class", type=int, default=200)
    smoke.add_argument("--chunk-size", type=int, default=100_000)
    smoke.add_argument("--seed", type=int, default=42)
    smoke.add_argument("--split-seed", type=int, default=123)
    smoke.add_argument("--epochs", type=int, default=2)
    smoke.add_argument("--batch-size", type=int, default=64)
    smoke.add_argument(
        "--output",
        default=None,
        help="Must be a new, non-existing run directory",
    )
    smoke.set_defaults(function=paper_smoke)
    paper_training = subparsers.add_parser(
        "paper-tabnet",
        help="Run validation-selected, three-seed paper-profile TabNet on a sampled CSV",
    )
    paper_training.add_argument("--dataset", required=True, help="Pre-sampled CIC or OD CSV; do not combine datasets")
    paper_training.add_argument("--dataset-name", required=True, help="Dataset identity recorded in the run")
    paper_training.add_argument("--label", default="Label")
    paper_training.add_argument("--mode", choices=("classical", "amplitude", "qatnet"), default="classical")
    paper_training.add_argument("--qubits", type=int, choices=(4, 5, 6), default=4)
    paper_training.add_argument("--split-seed", type=int, required=True)
    paper_training.add_argument("--output", required=True, help="New run directory; existing paths are rejected")
    paper_training.set_defaults(function=paper_tabnet)
    paired_training = subparsers.add_parser(
        "paired-tabnet",
        help="Run the fixed-architecture paired PCA/amplitude-probability pilot",
    )
    paired_training.add_argument("--dataset", required=True)
    paired_training.add_argument("--dataset-name", required=True)
    paired_training.add_argument("--label", default="Label")
    paired_training.add_argument("--qubits", type=int, choices=(4,), default=4)
    paired_training.add_argument("--split-seed", type=int, required=True)
    paired_training.add_argument("--output", required=True, help="New run directory; existing paths are rejected")
    paired_training.set_defaults(function=paired_tabnet)
    ablation = subparsers.add_parser(
        "representation-ablation",
        help="Run the fixed three-way Q4 PCA/normalized/probability representation ablation",
    )
    ablation.add_argument("--dataset", required=True)
    ablation.add_argument("--dataset-name", required=True)
    ablation.add_argument("--label", default="Label")
    ablation.add_argument("--qubits", type=int, choices=(4,), default=4)
    ablation.add_argument("--split-seed", type=int, required=True)
    ablation.add_argument("--output", required=True, help="New run directory; existing paths are rejected")
    ablation.set_defaults(
        function=paired_tabnet,
        representation_modes=("classical", "normalized_signed", "amplitude"),
    )
    exploration = subparsers.add_parser(
        "sample-explore",
        help="Create a bounded, per-label exploratory sample without paper-class mapping",
    )
    exploration.add_argument("--data-dir", default="cc")
    exploration.add_argument("--label", default="Label")
    exploration.add_argument("--samples-per-class", type=int, default=1000)
    exploration.add_argument("--chunk-size", type=int, default=100_000)
    exploration.add_argument("--seed", type=int, default=42)
    exploration.add_argument("--output", default=None, help="Must be a new run directory")
    exploration.set_defaults(function=explore_sample)
    pilot = subparsers.add_parser(
        "prepare-cic-pilot",
        help="Create a balanced five-class CIC dataset and leakage/provenance manifest; no training",
    )
    pilot.add_argument("--data-dir", default="cc")
    pilot.add_argument(
        "--source-files",
        nargs="+",
        default=[
            "02-14-2018.csv",
            "02-16-2018.csv",
            "02-28-2018.csv",
            "03-01-2018.csv",
            "03-02-2018.csv",
        ],
        help="Capture CSVs to sample; defaults to the five locally audited attack-bearing dates",
    )
    pilot.add_argument("--label", default="Label")
    pilot.add_argument("--samples-per-class", type=int, default=1000)
    pilot.add_argument("--chunk-size", type=int, default=50_000)
    pilot.add_argument("--seed", type=int, default=20261029)
    pilot.add_argument("--split-seed", type=int, default=101)
    pilot.add_argument(
        "--output",
        default="outputs/runs/cic-paper-pilot-20261029",
        help="New run directory; existing paths are rejected",
    )
    pilot.set_defaults(function=prepare_cic_pilot)
    integration = subparsers.add_parser(
        "quasar-qatnet-smoke",
        help="Train a small Q=4 QATNet model and replay CIC test flows through QUASAR mocks",
    )
    integration.add_argument("--sample-csv", required=True)
    integration.add_argument("--label", default="Label")
    integration.add_argument("--seed", type=int, default=42)
    integration.add_argument("--split-seed", type=int, default=123)
    integration.add_argument("--pretrain-samples", type=int, default=100)
    integration.add_argument("--pretrain-epochs", type=int, default=1)
    integration.add_argument("--epochs", type=int, default=2)
    integration.add_argument("--batch-size", type=int, default=32)
    integration.add_argument("--replay-flows", type=int, default=8)
    integration.add_argument("--output", default=None, help="Must be a new run directory")
    integration.set_defaults(function=qatnet_quasar_smoke)
    noisy = subparsers.add_parser("noise", help="Run the separate noisy amplitude-feature learner experiment")
    noisy.add_argument("--dataset", required=True)
    noisy.add_argument("--label", default="Label")
    noisy.add_argument("--qubits", type=int, choices=(4, 5, 6), default=4)
    noisy.add_argument("--shots", type=int, default=1024)
    noisy.add_argument("--depolarizing-probability", type=float, default=0.01)
    noisy.add_argument("--sample-rows", type=int)
    noisy.add_argument("--seed", type=int, default=42)
    noisy.add_argument("--data-kind", choices=("real", "synthetic", "unspecified"), default="unspecified")
    noisy.add_argument("--output", default="outputs/noise")
    noisy.set_defaults(function=noise_experiment)
    qkd = subparsers.add_parser("simulate-qkd", help="Generate explicitly synthetic QKD/KMS telemetry")
    qkd.add_argument("--steps", type=int, default=100)
    qkd.add_argument("--scenario", choices=sorted(QKDSimulator.SCENARIOS), default="normal")
    qkd.add_argument("--seed", type=int, default=42)
    qkd.add_argument("--initial-pool", type=float, default=100_000)
    qkd.add_argument("--csv", default="outputs/qkd/telemetry.csv")
    qkd.add_argument("--json", default="outputs/qkd/telemetry.json")
    qkd.set_defaults(function=simulate_qkd)
    demo = subparsers.add_parser("demo", help="Run deterministic synthetic QKD/SDN/QUASAR demonstration")
    demo.add_argument("--output", default="outputs/demo")
    demo.set_defaults(function=lambda args: logger.info(
        "Synthetic demo completed:\n%s", json.dumps(run_demo(args.output), indent=2, default=str)
    ))
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
