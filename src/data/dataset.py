from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import resource
import sys
import time
from typing import Iterable, Mapping, Sequence
import unicodedata
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


IDENTIFIER_PATTERN = re.compile(
    r"(^|[_\s-])(src|source|dst|destination)[_\s-]*(ip|ipv4|ipv6|port|address)"
    r"|(^|[_\s-])(flow.?id|timestamp|row.?id|index)([_\s-]|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DatasetReport:
    rows: int
    columns: int
    label_column: str
    class_counts: dict[str, int]
    missing_values: int
    infinite_values: int
    repeated_header_rows: int
    dropped_identifiers: tuple[str, ...]
    sampling_seed: int | None = None
    split_seed: int | None = None
    original_label_counts: dict[str, int] | None = None
    sampled_label_counts: dict[str, int] | None = None


@dataclass
class DataSplits:
    x_train: np.ndarray
    x_validation: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    class_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    report: DatasetReport
    preprocessor: "FlowPreprocessor"
    raw_x_test: pd.DataFrame
    split_class_counts: dict[str, dict[str, int]]


PAPER_CLASS_MAPPING = {
    "Benign": "Benign",
    "Bot": "Bot",
    "DoS attacks-Hulk": "DoS-Hulk",
    "FTP-BruteForce": "FTP-BruteForce",
    "Infilteration": "Infiltration",
}
PAPER_CLASSES = tuple(PAPER_CLASS_MAPPING.values())


@dataclass(frozen=True)
class SamplingReport:
    source_files: tuple[str, ...]
    source_row_counts: dict[str, int]
    original_label_counts: dict[str, int]
    normalized_label_counts: dict[str, int]
    sampled_label_counts: dict[str, int]
    repeated_header_rows: int
    missing_values: int
    infinite_values: int
    excluded_columns: tuple[str, ...]
    sampling_seed: int
    requested_per_class: int
    duplicate_policy: str = "No duplicate removal; repeated records are retained."


@dataclass
class SampledDataset:
    frame: pd.DataFrame
    report: SamplingReport


def normalize_label(value: object) -> str:
    if pd.isna(value):
        return "<MISSING>"
    return unicodedata.normalize("NFC", str(value)).strip()


def _normalize_sample_features(frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
    identifiers = {
        column for column in frame.columns
        if column != label_column and IDENTIFIER_PATTERN.search(str(column))
    }
    for column in frame.columns:
        if column == label_column or column in identifiers or not (
            pd.api.types.is_object_dtype(frame[column])
            or pd.api.types.is_string_dtype(frame[column])
        ):
            continue
        nonmissing = frame[column].notna()
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric[nonmissing].notna().all():
            frame[column] = numeric
        else:
            frame[column] = frame[column].map(
                lambda value: np.nan if pd.isna(value) else normalize_label(value)
            )
    return frame


def _header_mask(frame: pd.DataFrame) -> pd.Series:
    return frame.astype("string").eq(
        pd.Series(frame.columns, index=frame.columns, dtype="string"),
        axis="columns",
    ).all(axis=1)


def sample_csv_files(
    paths: Sequence[str | Path],
    samples_per_class: int,
    seed: int,
    label_column: str = "Label",
    label_mapping: Mapping[str, str] | None = None,
    target_classes: Sequence[str] | None = None,
    chunk_size: int = 100_000,
    require_samples_per_class: bool = True,
) -> SampledDataset:
    """Scan CSVs in bounded chunks and keep a seeded uniform reservoir per class."""
    if not paths:
        raise ValueError("At least one CSV path is required")
    if samples_per_class < 1 or chunk_size < 1:
        raise ValueError("samples_per_class and chunk_size must be positive")
    mapping = {normalize_label(k): normalize_label(v) for k, v in (label_mapping or {}).items()}
    if target_classes is not None and not mapping:
        raise ValueError("target_classes requires an explicit label_mapping")
    if target_classes is None and mapping:
        target_classes = tuple(dict.fromkeys(mapping.values()))
    targets = tuple(target_classes or ())
    if targets and len(set(targets)) != len(targets):
        raise ValueError("target_classes must be unique")

    rng = np.random.default_rng(seed)
    class_frames: dict[str, pd.DataFrame] = {}
    class_scores: dict[str, np.ndarray] = {}
    source_row_counts: dict[str, int] = {}
    original_counts: dict[str, int] = {}
    normalized_counts: dict[str, int] = {}
    repeated_headers = missing_values = infinite_values = 0
    all_columns: list[str] = []
    checked_paths = [Path(path) for path in paths]

    for path in checked_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Dataset CSV not found: {path}")
        header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
        if label_column not in header:
            raise ValueError(f"Label column {label_column!r} is missing from {path}")
        for column in header:
            if column not in all_columns:
                all_columns.append(column)
        source_row_counts[str(path)] = 0
        for chunk in pd.read_csv(
            path, chunksize=chunk_size, low_memory=False, encoding="utf-8-sig"
        ):
            if label_column not in chunk.columns:
                raise ValueError(f"Label column {label_column!r} is missing from {path}")
            repeated = _header_mask(chunk)
            repeated_headers += int(repeated.sum())
            chunk = chunk.loc[~repeated].copy()
            if chunk.empty:
                continue
            source_labels = chunk[label_column].map(
                lambda value: "<MISSING>" if pd.isna(value) else str(value)
            )
            labels = source_labels.map(normalize_label)
            chunk[label_column] = labels
            source_row_counts[str(path)] += len(chunk)
            for label, count in source_labels.value_counts().items():
                original_counts[label] = original_counts.get(label, 0) + int(count)
            for label, count in labels.value_counts().items():
                normalized_counts[label] = normalized_counts.get(label, 0) + int(count)
            numeric = chunk.drop(columns=[label_column]).select_dtypes(include=[np.number])
            if not numeric.empty:
                values = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
                missing_values += int(chunk.isna().sum().sum())
                infinite_values += int(np.isinf(values).sum())
            else:
                missing_values += int(chunk.isna().sum().sum())

            if mapping:
                mapped = labels.map(mapping)
                selected_targets = targets
            else:
                mapped = labels
                selected_targets = tuple(labels.unique()) if not targets else targets

            for target in selected_targets:
                mask = mapped.eq(target)
                count = int(mask.sum())
                if not count:
                    continue
                candidate = chunk.loc[mask].reindex(columns=all_columns)
                candidate[label_column] = target
                candidate_scores = rng.random(count)
                prior = class_frames.get(target)
                if prior is not None:
                    candidate = pd.concat([prior, candidate], ignore_index=True)
                    candidate_scores = np.concatenate([class_scores[target], candidate_scores])
                if len(candidate) > samples_per_class:
                    keep = np.argpartition(candidate_scores, samples_per_class - 1)[:samples_per_class]
                    candidate = candidate.iloc[keep]
                    candidate_scores = candidate_scores[keep]
                class_frames[target] = candidate.reset_index(drop=True)
                class_scores[target] = candidate_scores

    if not targets:
        targets = tuple(sorted(class_frames))
    absent = [
        name for name in targets
        if require_samples_per_class and len(class_frames.get(name, ())) < samples_per_class
    ]
    if absent:
        available = {
            target: int(sum(count for raw, count in normalized_counts.items()
                            if (mapping.get(raw, raw) == target)))
            for target in targets
        }
        raise ValueError(
            f"Insufficient source examples for requested classes {absent}; "
            f"available counts are {available}, requested {samples_per_class} per class"
        )
    sampled = pd.concat(
        [class_frames[name] for name in targets], ignore_index=True
    ).reindex(columns=all_columns)
    sampled = _normalize_sample_features(sampled, label_column)
    sampled_counts = {
        str(key): int(value)
        for key, value in sampled[label_column].value_counts().items()
    }
    identifiers = tuple(
        column for column in all_columns
        if column != label_column and IDENTIFIER_PATTERN.search(str(column))
    )
    report = SamplingReport(
        source_files=tuple(str(path) for path in checked_paths),
        source_row_counts=source_row_counts,
        original_label_counts=original_counts,
        normalized_label_counts=normalized_counts,
        sampled_label_counts=sampled_counts,
        repeated_header_rows=repeated_headers,
        missing_values=missing_values,
        infinite_values=infinite_values,
        excluded_columns=identifiers,
        sampling_seed=seed,
        requested_per_class=samples_per_class,
    )
    sampled.attrs["sampling_report"] = report
    return SampledDataset(sampled, report)


def load_csv(path: str | Path, label_column: str = "Label") -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {path}")
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise ValueError(f"Dataset CSV is empty: {path}")
    if label_column not in frame.columns:
        raise ValueError(f"Label column {label_column!r} is missing from {path}")
    return _remove_repeated_headers(frame)


def _remove_repeated_headers(frame: pd.DataFrame) -> pd.DataFrame:
    repeated = frame.astype("string").eq(
        pd.Series(frame.columns, index=frame.columns, dtype="string"),
        axis="columns",
    ).all(axis=1)
    removed = int(repeated.sum())
    previously_removed = int(frame.attrs.get("repeated_header_rows", 0))
    if removed:
        warnings.warn(
            f"Removed {removed} embedded repeated CSV header row(s)",
            RuntimeWarning,
            stacklevel=2,
        )
        frame = frame.loc[~repeated].copy()
    frame.attrs["repeated_header_rows"] = previously_removed + removed
    if frame.empty:
        raise ValueError("Dataset contains no data rows after repeated headers are removed")
    return frame


def inspect_dataset(frame: pd.DataFrame, label_column: str) -> DatasetReport:
    if label_column not in frame:
        raise ValueError(f"Label column {label_column!r} is missing")
    frame = _remove_repeated_headers(frame)
    labels = frame[label_column].astype("string").fillna("<MISSING>")
    numeric = frame.drop(columns=[label_column]).select_dtypes(include=[np.number])
    finite = np.isfinite(numeric.to_numpy(dtype=np.float64, na_value=np.nan))
    return DatasetReport(
        rows=len(frame),
        columns=len(frame.columns),
        label_column=label_column,
        class_counts={str(k): int(v) for k, v in labels.value_counts(dropna=False).items()},
        missing_values=int(frame.isna().sum().sum()),
        infinite_values=int(np.isinf(numeric.to_numpy(dtype=np.float64, na_value=np.nan)).sum()),
        repeated_header_rows=int(frame.attrs.get("repeated_header_rows", 0)),
        dropped_identifiers=tuple(str(c) for c in frame.columns if c != label_column and IDENTIFIER_PATTERN.search(str(c))),
    )


class FlowPreprocessor:
    """Train-fitted imputation, scaling, one-hot encoding and PCA."""

    def __init__(self, components: int, drop_columns: Iterable[str] = ()) -> None:
        if components < 1:
            raise ValueError("components must be positive")
        self.requested_components = components
        self.drop_columns = set(drop_columns)
        self.transformer: ColumnTransformer | None = None
        self.pca: PCA | None = None
        self.feature_names: tuple[str, ...] = ()
        self.input_feature_names: tuple[str, ...] = ()
        self.component_count = 0

    def _features(self, frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
        automatic = {str(c) for c in frame.columns if IDENTIFIER_PATTERN.search(str(c))}
        columns = [c for c in frame.columns if c != label_column and c not in automatic | self.drop_columns]
        if not columns:
            raise ValueError("No model features remain after identifier columns are removed")
        result = frame.loc[:, columns].copy()
        result = result.replace([np.inf, -np.inf], np.nan)
        return result

    def fit_transform(self, frame: pd.DataFrame, label_column: str) -> np.ndarray:
        features = self._features(frame, label_column)
        self.input_feature_names = tuple(str(column) for column in features.columns)
        numeric_columns = list(features.select_dtypes(include=[np.number, "bool"]).columns)
        categorical_columns = [column for column in features.columns if column not in numeric_columns]
        transformers = []
        if numeric_columns:
            transformers.append(
                ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_columns)
            )
        if categorical_columns:
            transformers.append(
                ("categorical", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]), categorical_columns)
            )
        self.transformer = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
        transformed = np.asarray(self.transformer.fit_transform(features), dtype=np.float32)
        self.feature_names = tuple(str(v) for v in self.transformer.get_feature_names_out())
        self.component_count = min(self.requested_components, transformed.shape[1], max(1, len(transformed) - 1))
        self.pca = PCA(n_components=self.component_count, random_state=0)
        reduced = self.pca.fit_transform(transformed).astype(np.float32)
        return self._pad_components(reduced)

    def transform(self, frame: pd.DataFrame, label_column: str) -> np.ndarray:
        if self.transformer is None or self.pca is None:
            raise RuntimeError("Preprocessor must be fitted on training data before transform")
        features = self._features(frame, label_column)
        missing = set(self.input_feature_names) - set(features.columns)
        if missing:
            raise ValueError(f"Input is missing required model features: {sorted(missing)}")
        features = features.loc[:, self.input_feature_names]
        transformed = np.asarray(self.transformer.transform(features), dtype=np.float32)
        reduced = self.pca.transform(transformed).astype(np.float32)
        return self._pad_components(reduced)

    def _pad_components(self, values: np.ndarray) -> np.ndarray:
        if values.shape[1] >= self.requested_components:
            return values[:, : self.requested_components]
        return np.pad(values, ((0, 0), (0, self.requested_components - values.shape[1])))


def prepare_splits(
    frame: pd.DataFrame,
    label_column: str = "Label",
    components: int = 16,
    seed: int = 42,
    sample_rows: int | None = None,
    drop_columns: Iterable[str] = (),
) -> DataSplits:
    if label_column not in frame:
        raise ValueError(f"Label column {label_column!r} is missing")
    if sample_rows is not None and sample_rows < 1:
        raise ValueError("sample_rows must be positive when supplied")
    working = _remove_repeated_headers(frame.copy())
    if sample_rows is not None and len(working) > sample_rows:
        if working[label_column].value_counts().min() < 10:
            raise ValueError("Each class needs at least ten examples for stratified 80/10/10 sampling")
        working, _ = train_test_split(
            working, train_size=sample_rows, random_state=seed, stratify=working[label_column]
        )
    labels = working[label_column].map(normalize_label)
    counts = labels.value_counts()
    if len(counts) < 2:
        raise ValueError("At least two label classes are required")
    if counts.min() < 10:
        raise ValueError("Each class needs at least ten examples for stratified 80/10/10 splitting")
    train_idx, val_idx, test_idx = _stratified_split_indices(labels, seed)
    train_frame = working.iloc[train_idx]
    validation_frame = working.iloc[val_idx]
    test_frame = working.iloc[test_idx]

    encoder = LabelEncoder().fit(labels)
    y_train = encoder.transform(labels.iloc[train_idx])
    y_validation = encoder.transform(labels.iloc[val_idx])
    y_test = encoder.transform(labels.iloc[test_idx])
    preprocessor = FlowPreprocessor(components, drop_columns)
    x_train = preprocessor.fit_transform(train_frame, label_column)
    x_validation = preprocessor.transform(validation_frame, label_column)
    x_test = preprocessor.transform(test_frame, label_column)
    report = inspect_dataset(working, label_column)
    sample_report = frame.attrs.get("sampling_report")
    report = DatasetReport(
        rows=report.rows,
        columns=report.columns,
        label_column=report.label_column,
        class_counts=report.class_counts,
        missing_values=report.missing_values,
        infinite_values=report.infinite_values,
        repeated_header_rows=report.repeated_header_rows,
        dropped_identifiers=tuple(sorted(set(report.dropped_identifiers) | set(drop_columns))),
        sampling_seed=sample_report.sampling_seed if sample_report else None,
        split_seed=seed,
        original_label_counts=sample_report.original_label_counts if sample_report else report.class_counts,
        sampled_label_counts=sample_report.sampled_label_counts if sample_report else report.class_counts,
    )
    split_class_counts = {
        split_name: {
            class_name: int(np.count_nonzero(split_labels == class_index))
            for class_index, class_name in enumerate(encoder.classes_)
        }
        for split_name, split_labels in (
            ("train", y_train), ("validation", y_validation), ("test", y_test)
        )
    }
    return DataSplits(
        x_train, x_validation, x_test,
        y_train.astype(np.int64), y_validation.astype(np.int64), y_test.astype(np.int64),
        tuple(str(c) for c in encoder.classes_), preprocessor.feature_names, report, preprocessor,
        test_frame.copy(), split_class_counts,
    )


def _stratified_split_indices(
    labels: pd.Series, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    train_idx, remainder_idx = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=labels
    )
    validation_idx, test_idx = train_test_split(
        remainder_idx,
        test_size=0.5,
        random_state=seed,
        stratify=labels.iloc[remainder_idx],
    )
    return train_idx, validation_idx, test_idx


def _duplicate_split_report(
    frame: pd.DataFrame,
    split_indices: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, int | str]:
    fingerprints = pd.util.hash_pandas_object(
        frame, index=False, categorize=True
    ).to_numpy(dtype=np.uint64)
    split_hashes = {}
    for name, indices in zip(("train", "validation", "test"), split_indices):
        order = np.argsort(fingerprints[indices], kind="stable")
        split_indices_sorted = indices[order]
        split_hashes[name] = (
            fingerprints[split_indices_sorted],
            split_indices_sorted,
        )
    overlaps = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        matched_rows = 0
        left_hashes, left_indices = split_hashes[left]
        right_hashes, right_indices = split_hashes[right]
        left_pos = right_pos = 0
        while left_pos < len(left_hashes) and right_pos < len(right_hashes):
            if left_hashes[left_pos] < right_hashes[right_pos]:
                left_pos += 1
                continue
            if right_hashes[right_pos] < left_hashes[left_pos]:
                right_pos += 1
                continue
            fingerprint = left_hashes[left_pos]
            left_end = left_pos + 1
            right_end = right_pos + 1
            while left_end < len(left_hashes) and left_hashes[left_end] == fingerprint:
                left_end += 1
            while right_end < len(right_hashes) and right_hashes[right_end] == fingerprint:
                right_end += 1
            unmatched_right = list(right_indices[right_pos:right_end])
            for left_index in left_indices[left_pos:left_end]:
                match_index = next(
                    (
                        offset for offset, right_index in enumerate(unmatched_right)
                        if frame.iloc[left_index].equals(frame.iloc[right_index])
                    ),
                    None,
                )
                if match_index is not None:
                    unmatched_right.pop(match_index)
                    matched_rows += 1
            left_pos = left_end
            right_pos = right_end
        overlaps[f"{left}_and_{right}_matching_rows"] = matched_rows
    return {
        **overlaps,
        "fingerprint": (
            "pandas 64-bit hash groups candidates; full-row equality is checked across "
            "all sampled columns, including label"
        ),
        "policy": "Report exact-row hash overlap only; retain all repeated traffic records.",
    }


def prepare_paper_dataset(
    paths: Sequence[str | Path],
    output_dir: str | Path,
    samples_per_class: int,
    sampling_seed: int,
    split_seed: int,
    label_column: str = "Label",
    chunk_size: int = 50_000,
) -> dict[str, object]:
    """Create a balanced pilot CSV and leakage/provenance manifest without training."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset run: {output}")
    started = time.perf_counter()
    sampled = sample_csv_files(
        paths,
        samples_per_class=samples_per_class,
        seed=sampling_seed,
        label_column=label_column,
        label_mapping=PAPER_CLASS_MAPPING,
        target_classes=PAPER_CLASSES,
        chunk_size=chunk_size,
    )
    splits = prepare_splits(
        sampled.frame,
        label_column=label_column,
        components=16,
        seed=split_seed,
    )
    label_values = sampled.frame[label_column].map(normalize_label)
    train_idx, validation_idx, test_idx = _stratified_split_indices(label_values, split_seed)
    duplicate_report = _duplicate_split_report(
        sampled.frame, (train_idx, validation_idx, test_idx)
    )
    numeric = sampled.frame.drop(columns=[label_column]).select_dtypes(include=[np.number])
    sampled_infinite = (
        int(np.isinf(numeric.to_numpy(dtype=np.float64, na_value=np.nan)).sum())
        if not numeric.empty else 0
    )
    manifest: dict[str, object] = {
        "experiment": "CIC-IDS2018 five-class balanced pilot dataset; no model training performed",
        "source_files": list(sampled.report.source_files),
        "source_row_counts_after_repeated_header_removal": sampled.report.source_row_counts,
        "paper_target_to_exact_source_label": {
            target: [source for source, mapped in PAPER_CLASS_MAPPING.items() if mapped == target]
            for target in PAPER_CLASSES
        },
        "explicit_aliases": {
            "DoS attacks-Hulk": "DoS-Hulk",
            "Infilteration": "Infiltration",
        },
        "sampling": {
            "method": "uniform seeded per-class reservoir sampling without replacement",
            "seed": sampling_seed,
            "requested_per_class": samples_per_class,
            "sampled_class_counts": sampled.report.sampled_label_counts,
            "original_source_label_counts": sampled.report.original_label_counts,
            "normalized_source_label_counts": sampled.report.normalized_label_counts,
            "duplicate_policy": sampled.report.duplicate_policy,
        },
        "split": {
            "method": "two-stage stratified 80/10/10",
            "seed": split_seed,
            "class_counts": splits.split_class_counts,
            "rows": {
                "train": len(splits.y_train),
                "validation": len(splits.y_validation),
                "test": len(splits.y_test),
            },
            "test_use": "held out; no fitting, sampling decisions, model selection, or threshold tuning",
            "model_selection": "not performed during dataset preparation",
        },
        "preprocessing": {
            "fit_partition": "train only",
            "validation_test_operation": "transform using the fitted training preprocessor",
            "missing_values": {
                "source_scan_count": sampled.report.missing_values,
                "pilot_count": int(sampled.frame.isna().sum().sum()),
                "handling": "numeric median and categorical most-frequent imputation fitted on training rows",
            },
            "infinite_values": {
                "source_scan_count_numeric": sampled.report.infinite_values,
                "pilot_count_numeric": sampled_infinite,
                "handling": "replace +/-infinity with missing before train-only imputation",
            },
            "requested_pca_components": 16,
            "fitted_pca_components": splits.preprocessor.component_count,
            "pca_fit_partition": "train only",
            "fitted_training_rows": len(splits.y_train),
        },
        "features": {
            "source_model_feature_columns": list(splits.preprocessor.input_feature_names),
            "transformed_feature_names_before_pca": list(splits.feature_names),
            "pca_output_columns": [
                f"PC{index + 1}" for index in range(splits.preprocessor.requested_components)
            ],
            "excluded_identifier_columns": list(sampled.report.excluded_columns),
        },
        "repeated_header_rows_removed": sampled.report.repeated_header_rows,
        "cross_split_duplicate_check": duplicate_report,
    }

    output.mkdir(parents=True, exist_ok=False)
    sampled.frame.to_csv(output / "sampled.csv", index=False)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_bytes = int(peak_rss if sys.platform == "darwin" else peak_rss * 1024)
    manifest["preparation_seconds"] = time.perf_counter() - started
    manifest["resource_measurements"] = {
        "process_peak_rss_bytes": peak_rss_bytes,
        "method": "resource.getrusage(RUSAGE_SELF).ru_maxrss; platform-normalized bytes",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
