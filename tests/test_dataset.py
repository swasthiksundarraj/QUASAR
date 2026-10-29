import numpy as np
import pandas as pd
import pytest

from data.dataset import (
    FlowPreprocessor,
    inspect_dataset,
    load_csv,
    normalize_label,
    prepare_paper_dataset,
    prepare_splits,
    sample_csv_files,
)


def sample_frame(count_per_class=30):
    rng = np.random.default_rng(4)
    rows = []
    for label in ("BENIGN", "Attack"):
        for index in range(count_per_class):
            rows.append({
                "Flow ID": f"{label}-{index}",
                "Source IP": f"10.0.0.{index}",
                "Destination Port": index,
                "duration": float(rng.normal(10 if label == "BENIGN" else 20, 2)),
                "packets": float(index + (100 if label == "Attack" else 0)),
                "category": "normal" if label == "BENIGN" else "suspicious",
                "Label": label,
            })
    return pd.DataFrame(rows)


def test_splits_are_stratified_and_identifier_columns_removed():
    splits = prepare_splits(sample_frame(), components=4, seed=3)
    assert splits.x_train.shape[1] == 4
    assert set(splits.y_train) == {0, 1}
    assert len(splits.x_train) + len(splits.x_validation) + len(splits.x_test) == 60
    assert splits.split_class_counts == {
        "train": {"BENIGN": 24, "Attack": 24},
        "validation": {"BENIGN": 3, "Attack": 3},
        "test": {"BENIGN": 3, "Attack": 3},
    }
    assert not any("Source IP" in name or "Flow ID" in name for name in splits.feature_names)


def test_preprocessor_requires_training_fit_and_reuses_train_transform():
    frame = sample_frame()
    processor = FlowPreprocessor(components=3)
    with pytest.raises(RuntimeError):
        processor.transform(frame, "Label")
    train = frame.iloc[:40]
    processor.fit_transform(train, "Label")
    output = processor.transform(frame.iloc[40:], "Label")
    assert output.shape == (20, 3)
    assert np.isfinite(output).all()


def test_bad_csv_and_too_small_strata_fail_explicitly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "missing.csv")
    with pytest.raises(ValueError, match="at least ten"):
        prepare_splits(sample_frame(count_per_class=2), components=2)


def test_load_csv_removes_and_reports_embedded_header_rows(tmp_path):
    frame = sample_frame(count_per_class=10)
    repeated_header = pd.DataFrame([{column: column for column in frame.columns}])
    path = tmp_path / "with-repeated-header.csv"
    pd.concat([frame, repeated_header], ignore_index=True).to_csv(path, index=False)

    with pytest.warns(RuntimeWarning, match="Removed 1 embedded repeated CSV header"):
        loaded = load_csv(path)

    report = inspect_dataset(loaded, "Label")
    assert len(loaded) == len(frame)
    assert report.rows == len(frame)
    assert report.repeated_header_rows == 1
    assert set(report.class_counts) == {"BENIGN", "Attack"}


def test_chunked_sampling_is_reproducible_across_different_schemas(tmp_path):
    first = sample_frame(count_per_class=40).rename(columns={"Source IP": "Src IP"})
    second = sample_frame(count_per_class=35).drop(columns=["Source IP", "Destination Port"])
    second["Label"] = second["Label"].str.replace("Attack", "Attack ")
    repeated = pd.DataFrame([{column: column for column in first.columns}])
    first_path, second_path = tmp_path / "first.csv", tmp_path / "second.csv"
    second["duration"] = second["duration"].map(lambda value: f"{value:.2f}")
    pd.concat([first, repeated], ignore_index=True).to_csv(first_path, index=False)
    second.to_csv(second_path, index=False)
    mapping = {"BENIGN": "Benign", "Attack": "Attack"}

    sample_a = sample_csv_files(
        [first_path, second_path], 20, 17, label_mapping=mapping,
        target_classes=("Benign", "Attack"), chunk_size=13,
    )
    sample_b = sample_csv_files(
        [first_path, second_path], 20, 17, label_mapping=mapping,
        target_classes=("Benign", "Attack"), chunk_size=13,
    )

    pd.testing.assert_frame_equal(sample_a.frame, sample_b.frame)
    assert sample_a.frame["Label"].value_counts().to_dict() == {"Benign": 20, "Attack": 20}
    assert "Src IP" in sample_a.frame and "Destination Port" in sample_a.frame
    assert pd.api.types.is_numeric_dtype(sample_a.frame["duration"])
    assert sample_a.report.repeated_header_rows == 1
    assert sample_a.report.source_row_counts[str(first_path)] == 80
    assert sample_a.report.original_label_counts.get("Attack ") == 35
    assert sample_a.report.normalized_label_counts["Attack"] == 75
    assert sample_a.report.sampling_seed == 17
    assert "Src IP" in sample_a.report.excluded_columns
    assert sample_a.report.duplicate_policy.startswith("No duplicate removal")


def test_sampler_reads_chunks_instead_of_materializing_source_files(tmp_path, monkeypatch):
    frame = sample_frame(count_per_class=20)
    path = tmp_path / "sample.csv"
    frame.to_csv(path, index=False)
    original_read_csv = pd.read_csv
    chunk_sizes = []

    def checked_read_csv(*args, **kwargs):
        if kwargs.get("chunksize") is not None:
            chunk_sizes.append(kwargs["chunksize"])
            return original_read_csv(*args, **kwargs)
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", checked_read_csv)
    sample_csv_files(
        [path], 5, 3, label_mapping={"BENIGN": "Benign", "Attack": "Attack"},
        target_classes=("Benign", "Attack"), chunk_size=7,
    )
    assert chunk_sizes == [7]


def test_label_mapping_normalizes_whitespace_and_rare_classes_fail(tmp_path):
    assert normalize_label("  Cafe\u0301 ") == "Café"
    frame = sample_frame(count_per_class=4)
    path = tmp_path / "rare.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Insufficient source examples"):
        sample_csv_files(
            [path], 5, 1, label_mapping={"BENIGN": "Benign", "Attack": "Attack"},
            target_classes=("Benign", "Attack"), chunk_size=3,
        )


def test_all_label_exploration_caps_common_classes_without_dropping_rare_labels(tmp_path):
    frame = sample_frame(count_per_class=4)
    frame.loc[0, "Label"] = "Rare "
    path = tmp_path / "all-labels.csv"
    frame.to_csv(path, index=False)

    sample = sample_csv_files(
        [path], 3, 5, chunk_size=2, require_samples_per_class=False
    )

    assert sample.report.sampled_label_counts == {
        "Attack": 3, "BENIGN": 3, "Rare": 1,
    }


def test_preprocessing_is_train_only_and_identifier_columns_never_enter_pca():
    frame = sample_frame(count_per_class=30)
    train = frame.iloc[:40].copy()
    held_out = frame.iloc[40:].copy()
    held_out["duration"] = 1_000_000_000
    processor = FlowPreprocessor(components=4)
    processor.fit_transform(train, "Label")

    numeric_pipeline = processor.transformer.named_transformers_["numeric"]
    fitted_mean = numeric_pipeline.named_steps["scaler"].mean_[0]
    np.testing.assert_allclose(fitted_mean, train["duration"].mean())
    output = processor.transform(held_out, "Label")
    assert output.shape == (20, 4)
    assert not any(
        identifier in name
        for name in processor.feature_names
        for identifier in ("Flow ID", "Source IP", "Destination Port")
    )


def test_prepared_paper_pilot_writes_compatible_dataset_and_manifest(tmp_path):
    source_labels = {
        "Benign": "Benign",
        "Bot": "Bot",
        "DoS attacks-Hulk": "DoS-Hulk",
        "FTP-BruteForce": "FTP-BruteForce",
        "Infilteration": "Infiltration",
    }
    source_paths = []
    for class_index, source_label in enumerate(source_labels):
        rows = []
        for index in range(30):
            rows.append({
                "Flow ID": f"flow-{class_index}-{index // 10}",
                "Source IP": f"192.0.2.{class_index}",
                "Dst Port": 1000 + class_index,
                "signal": float(class_index * 10 + index // 10),
                "Label": source_label,
            })
        path = tmp_path / f"{class_index}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        source_paths.append(path)

    output = tmp_path / "pilot"
    manifest = prepare_paper_dataset(
        source_paths,
        output_dir=output,
        samples_per_class=20,
        sampling_seed=57,
        split_seed=101,
        chunk_size=7,
    )

    assert manifest["sampling"]["sampled_class_counts"] == {
        "Benign": 20,
        "Bot": 20,
        "DoS-Hulk": 20,
        "FTP-BruteForce": 20,
        "Infiltration": 20,
    }
    assert manifest["split"]["class_counts"] == {
        "train": {
            "Benign": 16, "Bot": 16, "DoS-Hulk": 16,
            "FTP-BruteForce": 16, "Infiltration": 16,
        },
        "validation": {
            "Benign": 2, "Bot": 2, "DoS-Hulk": 2,
            "FTP-BruteForce": 2, "Infiltration": 2,
        },
        "test": {
            "Benign": 2, "Bot": 2, "DoS-Hulk": 2,
            "FTP-BruteForce": 2, "Infiltration": 2,
        },
    }
    assert manifest["explicit_aliases"] == {
        "DoS attacks-Hulk": "DoS-Hulk",
        "Infilteration": "Infiltration",
    }
    assert manifest["preprocessing"]["fit_partition"] == "train only"
    assert manifest["preprocessing"]["fitted_training_rows"] == 80
    assert manifest["preparation_seconds"] >= 0
    assert manifest["resource_measurements"]["process_peak_rss_bytes"] > 0
    assert manifest["split"]["model_selection"] == "not performed during dataset preparation"
    assert manifest["features"]["excluded_identifier_columns"] == [
        "Flow ID", "Source IP", "Dst Port",
    ]
    assert manifest["features"]["pca_output_columns"] == [
        f"PC{index}" for index in range(1, 17)
    ]
    assert manifest["cross_split_duplicate_check"]["train_and_test_matching_rows"] > 0
    assert "retain all repeated traffic records" in manifest["cross_split_duplicate_check"]["policy"]
    assert (output / "sampled.csv").is_file()
    assert (output / "manifest.json").is_file()

    compatible = prepare_splits(
        load_csv(output / "sampled.csv"), components=16, seed=101
    )
    assert compatible.split_class_counts == manifest["split"]["class_counts"]
    assert compatible.x_train.shape[1] == 16
    assert not set(("Flow ID", "Source IP", "Dst Port")) & set(
        compatible.preprocessor.input_feature_names
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_paper_dataset(
            source_paths, output, 20, 57, 101, chunk_size=7
        )
