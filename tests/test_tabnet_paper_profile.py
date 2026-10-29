import json

import numpy as np
import torch

from qatnet.training import (
    PaperTabNetProfile,
    TabNetCandidate,
    run_paired_representation_comparison,
    run_paper_tabnet_workflow,
    select_paper_tabnet_candidate,
)


class RecordingTabNet:
    def __init__(self, seed, candidate, profile, records):
        self.seed = seed
        self.candidate = candidate
        self.profile = profile
        self.records = records
        self.best_epoch = 2
        self.history = RecordingHistory()
        self.records.append(self)

    def fit(self, X_train, y_train, **kwargs):
        self.X_train = np.asarray(X_train).copy()
        self.y_train = np.asarray(y_train).copy()
        self.fit_kwargs = kwargs

    def predict_proba(self, values):
        labels = (np.asarray(values)[:, 0] > 0.5).astype(int)
        probabilities = np.full((len(labels), 2), 0.1, dtype=np.float64)
        probabilities[np.arange(len(labels)), labels] = 0.9
        return probabilities


class RecordingHistory:
    history = {
        "loss": [0.8, 0.6, 0.4],
        "validation_logloss": [0.9, 0.7, 0.5],
        "validation_accuracy": [0.5, 0.75, 1.0],
    }


def _recording_factory(records):
    return lambda seed, candidate, profile: RecordingTabNet(
        seed, candidate, profile, records
    )


def test_paper_tabnet_profile_matches_verified_training_settings():
    profile = PaperTabNetProfile()

    assert profile.max_epochs == 60
    assert profile.patience == 15
    assert profile.batch_size == 4096
    assert profile.virtual_batch_size == 512
    assert profile.selection_seed == 101
    assert profile.model_seeds == (101, 202, 402)


def test_validation_selection_uses_training_only_class_weights_and_adam_settings():
    records = []
    profile = PaperTabNetProfile(candidates=(
        TabNetCandidate(8, 8, 3, "sparsemax"),
    ))
    x_train = np.array([[0.0], [0.1], [0.2], [1.0]], dtype=np.float32)
    y_train = np.array([0, 0, 0, 1])
    x_validation = np.array([[0.0], [1.0], [1.0], [1.0]], dtype=np.float32)
    y_validation = np.array([0, 1, 1, 1])

    selected = select_paper_tabnet_candidate(
        x_train, y_train, x_validation, y_validation, ("benign", "attack"),
        profile=profile, model_factory=_recording_factory(records),
    )

    assert selected["selection_seed"] == 101
    assert selected["selection_metric"] == "validation_macro_f1"
    np.testing.assert_allclose(selected["class_weights"], [2 / 3, 2.0])
    model = records[0]
    assert isinstance(model.fit_kwargs["loss_fn"], torch.nn.CrossEntropyLoss)
    np.testing.assert_allclose(model.fit_kwargs["loss_fn"].weight.numpy(), [2 / 3, 2.0])
    assert model.fit_kwargs["weights"] == 0
    assert model.fit_kwargs["max_epochs"] == 60
    assert model.fit_kwargs["patience"] == 15
    assert model.fit_kwargs["batch_size"] == 4096
    assert model.fit_kwargs["virtual_batch_size"] == 512
    assert model.fit_kwargs["eval_set"][0][1] is not y_train
    np.testing.assert_array_equal(model.fit_kwargs["eval_set"][0][1], y_validation)


def test_three_seed_refit_keeps_split_seed_and_test_out_of_training(tmp_path):
    records = []
    profile = PaperTabNetProfile(candidates=(
        TabNetCandidate(8, 8, 3, "sparsemax"),
    ))
    x_train = np.array([[0.0], [0.1], [0.2], [1.0]], dtype=np.float32)
    y_train = np.array([0, 0, 0, 1])
    x_validation = np.array([[0.0], [1.0]], dtype=np.float32)
    y_validation = np.array([0, 1])
    x_test = np.array([[99.0], [0.0]], dtype=np.float32)
    y_test = np.array([1, 0])

    report = run_paper_tabnet_workflow(
        x_train, y_train, x_validation, y_validation, x_test, y_test,
        ("benign", "attack"),
        split_seed=777,
        output_dir=tmp_path / "paper-tabnet-run",
        dataset_name="CIC-IDS2018-test-fixture",
        profile=profile,
        model_factory=_recording_factory(records),
    )

    assert report["split_seed"] == 777
    assert report["model_seeds"] == [101, 202, 402]
    assert [item["model_seed"] for item in report["per_seed"]] == [101, 202, 402]
    assert report["refit"]["training_rows"] == len(y_train) + len(y_validation)
    assert report["refit"]["class_weight_source"] == "original training partition only"
    assert report["refit"]["epochs"] == 3

    selection_model, *refit_models = records
    for model in refit_models:
        assert model.seed in (101, 202, 402)
        assert model.X_train.shape[0] == len(y_train) + len(y_validation)
        assert 99.0 not in model.X_train
        np.testing.assert_array_equal(
            model.X_train[:, 0],
            np.concatenate((x_train[:, 0], x_validation[:, 0])),
        )
        assert model.fit_kwargs["eval_set"] == []
        assert model.fit_kwargs["weights"] == 0
        assert model.fit_kwargs["max_epochs"] == 3
        np.testing.assert_allclose(model.fit_kwargs["loss_fn"].weight.numpy(), [2 / 3, 2.0])
    assert selection_model.seed == 101
    assert 99.0 not in selection_model.X_train

    saved = json.loads((tmp_path / "paper-tabnet-run" / "summary.json").read_text())
    assert saved["dataset"] == "CIC-IDS2018-test-fixture"
    assert saved["aggregate_metrics"]["accuracy"]["n"] == 3
    for seed in (101, 202, 402):
        assert (tmp_path / "paper-tabnet-run" / f"seed-{seed}" / "metrics.json").is_file()


def test_paper_profile_rejects_virtual_batch_not_smaller_than_batch():
    try:
        PaperTabNetProfile(batch_size=512, virtual_batch_size=512)
    except ValueError as exc:
        assert "smaller" in str(exc)
    else:
        raise AssertionError("invalid virtual batch size should fail")


def test_paired_comparison_uses_identical_config_and_never_fits_on_test(tmp_path):
    records = []
    profile = PaperTabNetProfile(
        model_seeds=(101, 202, 402),
        candidates=(TabNetCandidate(8, 8, 3, "sparsemax"),),
    )
    x_train = np.array([
        [-1.0, 0.0], [-0.8, 0.2], [0.8, 0.0], [1.0, 0.2],
    ], dtype=np.float32)
    y_train = np.array([0, 0, 1, 1])
    x_validation = np.array([[-0.9, 0.1], [0.9, 0.1]], dtype=np.float32)
    y_validation = np.array([0, 1])
    x_test = np.array([[99.0, 99.0], [-99.0, -99.0]], dtype=np.float32)
    y_test = np.array([1, 0])
    candidate = TabNetCandidate(8, 8, 3, "sparsemax")

    report = run_paired_representation_comparison(
        x_train, y_train, x_validation, y_validation, x_test, y_test,
        ("benign", "attack"),
        split_seed=101,
        output_dir=tmp_path / "paired",
        dataset_name="pilot-fixture",
        candidate=candidate,
        refit_epochs=57,
        profile=profile,
        model_factory=_recording_factory(records),
    )

    assert report["model_seeds"] == [101, 202, 402]
    assert report["training_class_weights"] == [1.0, 1.0]
    assert len(records) == 12
    for model in records:
        assert model.candidate == candidate
        assert 99.0 not in model.X_train
        assert -99.0 not in model.X_train
        assert model.fit_kwargs["batch_size"] == 4096
        assert model.fit_kwargs["virtual_batch_size"] == 512
        np.testing.assert_allclose(
            model.fit_kwargs["loss_fn"].weight.numpy(), [1.0, 1.0]
        )
    selection_models = [records[index] for index in (0, 2, 4, 6, 8, 10)]
    refit_models = [records[index] for index in (1, 3, 5, 7, 9, 11)]
    for model in selection_models:
        assert model.fit_kwargs["max_epochs"] == 60
        assert model.fit_kwargs["patience"] == 15
        assert model.fit_kwargs["eval_metric"] == ["logloss", "accuracy"]
        assert len(model.fit_kwargs["eval_set"]) == 1
        np.testing.assert_array_equal(model.fit_kwargs["eval_set"][0][1], y_validation)
    for model in refit_models:
        assert model.fit_kwargs["max_epochs"] == 57
        assert model.fit_kwargs["patience"] == 0
        assert model.fit_kwargs["eval_set"] == []
        assert len(model.X_train) == len(y_train) + len(y_validation)
    classical_selection = records[0]
    amplitude_selection = records[6]
    np.testing.assert_array_equal(classical_selection.X_train, x_train)
    np.testing.assert_allclose(
        amplitude_selection.X_train,
        (x_train / np.linalg.norm(x_train, axis=1, keepdims=True)) ** 2,
    )
    assert report["aggregates"]["classical"][
        "test_metrics_after_fixed_train_plus_validation_refit"
    ]["accuracy"]["n"] == 3
    assert report["aggregates"]["classical"][
        "test_metrics_after_fixed_train_plus_validation_refit"
    ]["macro_precision"]["n"] == 3
    assert (
        tmp_path / "paired" / "classical" / "seed-101" / "selection-learning-curves.csv"
    ).is_file()
    assert (tmp_path / "paired" / "amplitude" / "test-predictions.csv").is_file()
    distributions = json.loads(
        (tmp_path / "paired" / "amplitude" / "predicted-class-distributions.json").read_text()
    )
    assert sum(distributions["101"]["test_after_fixed_refit"].values()) == len(y_test)
    assert (tmp_path / "paired" / "summary.json").is_file()


def test_paired_comparison_rejects_mismatched_partition_shapes(tmp_path):
    profile = PaperTabNetProfile(
        model_seeds=(101,),
        candidates=(TabNetCandidate(8, 8, 3, "sparsemax"),),
    )
    try:
        run_paired_representation_comparison(
            np.ones((4, 2)), np.array([0, 0, 1, 1]),
            np.ones((2, 3)), np.array([0, 1]),
            np.ones((2, 2)), np.array([0, 1]),
            ("benign", "attack"), 101, tmp_path / "bad", "pilot",
            TabNetCandidate(8, 8, 3, "sparsemax"), 57, profile=profile,
            model_factory=_recording_factory([]),
        )
    except ValueError as exc:
        assert "matching widths" in str(exc)
    else:
        raise AssertionError("mismatched feature widths should fail")
    assert not (tmp_path / "bad").exists()


def test_three_way_representation_ablation_transforms_and_training_are_paired(tmp_path):
    records = []
    candidate = TabNetCandidate(8, 8, 3, "sparsemax")
    profile = PaperTabNetProfile(
        model_seeds=(101, 202, 402),
        candidates=(candidate,),
    )
    x_train = np.array([
        [3.0, 4.0], [-3.0, 4.0], [1.0, -1.0], [-1.0, -1.0],
    ], dtype=np.float32)
    y_train = np.array([0, 0, 1, 1])
    x_validation = np.array([[6.0, 8.0], [-6.0, 8.0]], dtype=np.float32)
    y_validation = np.array([0, 1])
    x_test = np.array([[88.0, 88.0], [-88.0, -88.0]], dtype=np.float32)
    y_test = np.array([1, 0])
    dataset_manifest = {"source_csv_sha256": "fixture-hash", "split_seed": 101}

    report = run_paired_representation_comparison(
        x_train, y_train, x_validation, y_validation, x_test, y_test,
        ("benign", "attack"),
        split_seed=101,
        output_dir=tmp_path / "three-way",
        dataset_name="pilot-fixture",
        candidate=candidate,
        refit_epochs=57,
        profile=profile,
        model_factory=_recording_factory(records),
        representation_modes=("classical", "normalized_signed", "amplitude"),
        dataset_manifest=dataset_manifest,
    )

    assert list(report["per_representation_per_seed"]) == [
        "classical", "normalized_signed", "amplitude",
    ]
    assert report["model_seeds"] == [101, 202, 402]
    assert report["dataset_manifest"] == dataset_manifest
    expected_normalized = x_train / np.linalg.norm(x_train, axis=1, keepdims=True)
    expected_probabilities = np.square(expected_normalized)
    for mode, start in (("classical", 0), ("normalized_signed", 6), ("amplitude", 12)):
        selection_models = [records[start + i] for i in (0, 2, 4)]
        refit_models = [records[start + i] for i in (1, 3, 5)]
        for model in selection_models + refit_models:
            assert model.candidate == candidate
            assert model.fit_kwargs["batch_size"] == 4096
            assert model.fit_kwargs["virtual_batch_size"] == 512
            assert model.fit_kwargs["weights"] == 0
            assert model.fit_kwargs["max_epochs"] == (
                60 if model in selection_models else 57
            )
            assert 88.0 not in model.X_train and -88.0 not in model.X_train
            np.testing.assert_allclose(
                model.fit_kwargs["loss_fn"].weight.numpy(), [1.0, 1.0]
            )
        if mode == "classical":
            np.testing.assert_array_equal(selection_models[0].X_train, x_train)
        elif mode == "normalized_signed":
            np.testing.assert_allclose(
                selection_models[0].X_train, expected_normalized
            )
        else:
            np.testing.assert_allclose(
                selection_models[0].X_train, expected_probabilities
            )
        transform = json.loads(
            (tmp_path / "three-way" / mode / "transformation.json").read_text()
        )
        assert transform["representation"] == mode
        assert (tmp_path / "three-way" / mode / "dataset-manifest.json").is_file()
        assert (
            tmp_path / "three-way" / mode / "seed-101" / "dataset-manifest.json"
        ).is_file()
        assert (
            tmp_path / "three-way" / mode / "seed-101" / "classifier.joblib"
        ).is_file()
