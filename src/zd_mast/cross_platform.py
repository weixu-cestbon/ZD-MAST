"""Like-for-like ZD-MAST cross-platform analysis primitives.

This module implements the major-revision contract in which one Site A model,
calibrator, and set of validation-derived thresholds are applied unchanged to
Site A and Site B test cohorts represented in the same peak-presence space.
Target labels are used only for final evaluation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .metrics import (
    expected_calibration_error,
    threshold_at_min_sensitivity,
    threshold_at_min_specificity,
    youden_threshold,
)
from .modeling import (
    N_FEATURES,
    PublicProtocolData,
    apply_platt,
    calibration_slope_intercept,
    fit_lightgbm,
    fit_platt,
    matrix_rows,
    out_of_fold_predictions,
    predict,
    tune_with_frozen_folds,
)


SITE_A = "ZD-MAST-A"
SITE_B = "ZD-MAST-B"
DEFAULT_SEED = 20260815
DEFAULT_BOOTSTRAP_COUNT = 2000
MAX_METADATA_HASH_BYTES = 512 * 1024 * 1024

TASK_IDS: tuple[str, ...] = (
    "sa_oxa",
    "sa_lvx",
    "sa_gen",
    "kp_fep",
    "kp_cro",
    "kp_caz",
    "kp_cip",
    "ec_cro",
    "ec_cip",
    "ec_fep",
)

THRESHOLD_NAMES: tuple[str, ...] = ("youden", "sensitivity90", "specificity90")
THRESHOLD_METRIC_NAMES: tuple[str, ...] = (
    "threshold",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "false_susceptible_rate",
    "false_resistant_rate",
)

COHORT_WINDOWS: Mapping[str, tuple[str, str, str]] = {
    "site_a_test_patient_disjoint": ("2026-03-01", "2026-06-09", SITE_A),
    "site_a_test_all_samples": ("2026-03-01", "2026-06-09", SITE_A),
    "site_b_primary": ("2026-03-01", "2026-06-09", SITE_B),
    "site_b_full_period_sensitivity": ("2026-01-01", "2026-07-29", SITE_B),
}


@dataclass(frozen=True)
class SupportClassification:
    """Prespecified sample/class support classification."""

    status: str
    reason: str
    eligible_for_discrimination: bool


@dataclass(frozen=True)
class GuardedPlattCalibration:
    """Platt model plus an explicit non-positive-slope safety decision."""

    model: LogisticRegression | None
    slope: float
    intercept: float
    status: str
    reason: str

    @property
    def valid(self) -> bool:
        return self.status == "ok" and self.model is not None

    def apply(self, probability: np.ndarray) -> np.ndarray | None:
        """Apply calibration only when its fitted slope preserves ranking."""

        if not self.valid:
            return None
        return apply_platt(self.model, np.asarray(probability, dtype=float))


@dataclass(frozen=True)
class AnalysisInputs:
    """Validated public release inputs plus a caller-supplied target date bridge."""

    feature_root: Path
    matrices: Mapping[str, np.ndarray]
    sample_metadata: pd.DataFrame
    labels: pd.DataFrame
    groups: pd.DataFrame
    splits: pd.DataFrame
    folds: pd.DataFrame
    target_dates: pd.DataFrame
    target_date_source_column: str
    input_paths: Mapping[str, Path]
    matrix_audit: pd.DataFrame


@dataclass(frozen=True)
class TaskCohorts:
    """Source development/folds and explicit all-sample/purged cohorts."""

    task_id: str
    source_development: pd.DataFrame
    source_test_patient_disjoint: pd.DataFrame
    source_test_all_samples: pd.DataFrame
    source_folds: list[tuple[pd.DataFrame, pd.DataFrame, str]]
    source_test_purge_audit: pd.DataFrame
    fold_purge_audit: pd.DataFrame
    target_primary: pd.DataFrame
    target_full_period: pd.DataFrame

    @property
    def source_test(self) -> pd.DataFrame:
        """Compatibility alias: strict patient-disjoint Site A test."""

        return self.source_test_patient_disjoint


@dataclass(frozen=True)
class FittedSourceTask:
    """One frozen Site A task model and its development-only decisions."""

    task_id: str
    model: Any
    calibrator: GuardedPlattCalibration
    thresholds: Mapping[str, float]
    hyperparameters: Mapping[str, Any]
    tuning: pd.DataFrame
    oof: pd.DataFrame
    seed: int


def deterministic_seed(seed: int, *parts: object) -> int:
    """Derive a stable child seed without depending on process hash state."""

    token = "|".join(str(part) for part in parts).encode("utf-8")
    return int(seed + int(hashlib.sha256(token).hexdigest()[:8], 16) % 1_000_000)


def validate_task_ids(task_ids: Sequence[str]) -> tuple[str, ...]:
    """Validate and return task IDs in the frozen public-panel order."""

    requested = tuple(dict.fromkeys(str(task).strip() for task in task_ids if str(task).strip()))
    unknown = set(requested) - set(TASK_IDS)
    if unknown:
        raise ValueError(f"Unknown task IDs: {sorted(unknown)}")
    if not requested:
        raise ValueError("At least one task ID is required")
    requested_set = set(requested)
    return tuple(task for task in TASK_IDS if task in requested_set)


def protocol_config(
    task_ids: Sequence[str] = TASK_IDS,
    *,
    seed: int = DEFAULT_SEED,
    threads: int = 4,
    bootstrap_count: int = DEFAULT_BOOTSTRAP_COUNT,
) -> dict[str, object]:
    """Return the serialized analysis contract used by the CLI manifest."""

    return {
        "analysis_id": "cross_platform_like_for_like",
        "analysis_version": "major-revision-v1",
        "feature_representation": {
            "name": "peak_presence6000",
            "mz_range_da": [2000, 20000],
            "bin_width_da": 3,
            "n_features": 6000,
            "values": "binary_peak_presence",
        },
        "endpoint": "historical_S_vs_IR",
        "task_ids": list(validate_task_ids(task_ids)),
        "source_development": {
            "site": SITE_A,
            "window": ["2025-07-01", "2026-02-28"],
            "split_source": "public_frozen_local_temporal_current_workflow_protocol_b_train",
            "tuning_calibration_thresholds": "rolling_origin_oof_only",
        },
        "evaluation": {
            "site_a_test_patient_disjoint": ["2026-03-01", "2026-06-09"],
            "site_a_test_all_samples": ["2026-03-01", "2026-06-09"],
            "site_b_primary": ["2026-03-01", "2026-06-09"],
            "site_b_full_period_sensitivity": ["2026-01-01", "2026-07-29"],
        },
        "patient_disjoint_protocol": {
            "development_membership_unchanged": True,
            "fold_training_unchanged": True,
            "support_requirements": {
                "site_a_test_patient_disjoint": {
                    "minimum_total": 100,
                    "minimum_class": 20,
                },
                "rolling_origin_fold_training": {
                    "minimum_total": 20,
                    "minimum_class": 10,
                },
                "rolling_origin_fold_validation_after_purge": {
                    "minimum_total": 10,
                    "minimum_class": 5,
                },
            },
            "fold_validation_purge": [
                "remove_nonmissing_patient_clusters_seen_in_fold_training",
                "remove_missing_patient_cluster_rows",
            ],
            "site_a_test_purge": [
                "remove_nonmissing_patient_clusters_seen_in_development",
                "remove_missing_patient_cluster_rows",
            ],
            "primary_delta_source_cohort": "site_a_test_patient_disjoint",
            "all_sample_site_a_test_role": "sensitivity",
        },
        "model": "lightgbm",
        "bootstrap": {
            "unit": "patient_cluster_with_sample_fallback",
            "replicates": bootstrap_count,
            "interval": "percentile_95",
            "site_delta": "independent_site_wise_cluster_bootstrap",
        },
        "runtime": {"seed": seed, "threads": threads},
        "no_target_label_use": True,
    }


def classify_support(
    total_n: int,
    positive_n: int,
    negative_n: int,
    *,
    minimum_total: int = 100,
    minimum_class: int = 20,
) -> SupportClassification:
    """Classify support without dropping underpowered tasks from the audit."""

    if min(total_n, positive_n, negative_n) < 0 or positive_n + negative_n != total_n:
        raise ValueError("Inconsistent class counts")
    if total_n == 0:
        return SupportClassification("insufficient", "no_samples", False)
    if positive_n == 0 or negative_n == 0:
        return SupportClassification("insufficient", "single_class", False)
    if total_n >= minimum_total and min(positive_n, negative_n) >= minimum_class:
        return SupportClassification("adequate", "", True)
    reasons: list[str] = []
    if total_n < minimum_total:
        reasons.append(f"n<{minimum_total}")
    if min(positive_n, negative_n) < minimum_class:
        reasons.append(f"min_class<{minimum_class}")
    return SupportClassification(
        "exploratory_or_insufficient",
        ";".join(reasons),
        True,
    )


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt", ".tsv"}:
        separator = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=separator)
    raise ValueError(f"Unsupported table format: {path.name}")


def normalize_target_date_table(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Normalize a one-row-per-sample target date/group bridge."""

    if "public_sample_id" not in frame:
        raise ValueError("Target-date table is missing public_sample_id")
    if frame["public_sample_id"].isna().any():
        raise ValueError("Target-date table contains missing public_sample_id")
    if frame["public_sample_id"].astype(str).duplicated().any():
        count = int(frame["public_sample_id"].astype(str).duplicated(keep=False).sum())
        raise ValueError(f"Target-date table contains duplicate public_sample_id rows: {count}")
    date_column = next(
        (column for column in ("collection_date", "accept_datetime") if column in frame),
        None,
    )
    if date_column is None:
        raise ValueError("Target-date table requires collection_date or accept_datetime")
    columns = ["public_sample_id", date_column]
    if "public_patient_cluster_id" in frame:
        columns.append("public_patient_cluster_id")
    output = frame[columns].copy()
    output["public_sample_id"] = output["public_sample_id"].astype(str)
    parsed = pd.to_datetime(output[date_column], errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Unparseable or missing {date_column} values: {int(parsed.isna().sum())}")
    output["target_date"] = parsed.dt.normalize()
    final_columns = ["public_sample_id", "target_date"]
    if "public_patient_cluster_id" in output:
        output["public_patient_cluster_id"] = (
            output["public_patient_cluster_id"].astype("string").str.strip()
        )
        final_columns.append("public_patient_cluster_id")
    return output[final_columns], date_column


def load_target_date_table(path: Path) -> tuple[pd.DataFrame, str]:
    """Load CSV/Parquet target dates without retaining the input path downstream."""

    if not path.is_file():
        raise FileNotFoundError(path)
    return normalize_target_date_table(_read_table(path))


def filter_date_window(
    frame: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    date_column: str = "target_date",
) -> pd.DataFrame:
    """Filter an inclusive calendar-date window and preserve deterministic order."""

    if date_column not in frame:
        raise ValueError(f"Missing date column: {date_column}")
    dated = frame.copy()
    dated[date_column] = pd.to_datetime(dated[date_column], errors="raise").dt.normalize()
    lower, upper = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if lower > upper:
        raise ValueError("Date-window start is after end")
    selected = dated.loc[dated[date_column].between(lower, upper, inclusive="both")].copy()
    order = [date_column]
    if "public_sample_id" in selected:
        order.append("public_sample_id")
    return selected.sort_values(order, kind="stable").reset_index(drop=True)


def attach_target_dates(frame: pd.DataFrame, target_dates: pd.DataFrame) -> pd.DataFrame:
    """Attach dates without duplicating patient-group columns from the bridge."""

    _require_columns(frame, ["public_sample_id"], "target cohort")
    _require_columns(target_dates, ["public_sample_id", "target_date"], "target dates")
    return frame.merge(
        target_dates[["public_sample_id", "target_date"]],
        on="public_sample_id",
        how="left",
        validate="one_to_one",
    )


def reject_duplicate_labels(
    labels: pd.DataFrame,
    *,
    key: Sequence[str] = ("site_id", "public_sample_id", "task_id"),
) -> None:
    """Reject ambiguous duplicate site/sample/task outcomes."""

    missing = set(key) - set(labels.columns)
    if missing:
        raise ValueError(f"Label table missing key columns: {sorted(missing)}")
    duplicate = labels.duplicated(list(key), keep=False)
    if duplicate.any():
        preview = labels.loc[duplicate, list(key)].head(5).to_dict("records")
        raise ValueError(
            f"Duplicate site/sample/task labels detected: n={int(duplicate.sum())}; first={preview}"
        )


def reject_source_overlap(
    development: pd.DataFrame,
    test: pd.DataFrame,
    *,
    sample_column: str = "public_sample_id",
) -> None:
    """Reject any source sample assigned to both development and test."""

    if sample_column not in development or sample_column not in test:
        raise ValueError(f"Missing source overlap key: {sample_column}")
    overlap = set(development[sample_column].astype(str)) & set(test[sample_column].astype(str))
    if overlap:
        raise ValueError(
            f"Source development/test overlap detected: n={len(overlap)}; first={sorted(overlap)[:5]}"
        )


def reject_patient_cluster_overlap(
    development: pd.DataFrame,
    test: pd.DataFrame,
    *,
    group_column: str = "public_patient_cluster_id",
) -> None:
    """Reject patient clusters split across source development and test.

    Public releases may omit patient groups. In that case sample-level overlap
    remains the enforced gate and this function is a no-op.
    """

    if group_column not in development or group_column not in test:
        return
    development_groups = set(development[group_column].dropna().astype(str))
    test_groups = set(test[group_column].dropna().astype(str))
    overlap = development_groups & test_groups
    if overlap:
        raise ValueError(
            "Source development/test patient-cluster overlap detected: "
            f"n={len(overlap)}; first={sorted(overlap)[:5]}"
        )


def _present_patient_cluster_mask(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return rows with a usable, nonblank patient-cluster identifier."""

    if column not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame[column].astype("string")
    return values.notna() & values.str.strip().ne("")


def _class_support_record(
    frame: pd.DataFrame,
    *,
    minimum_total: int,
    minimum_class: int,
) -> dict[str, object]:
    """Summarize class support for an audit row without dropping the cohort."""

    positive_n = int(frame["y"].sum()) if len(frame) else 0
    negative_n = int(frame["y"].eq(0).sum()) if len(frame) else 0
    support = classify_support(
        len(frame),
        positive_n,
        negative_n,
        minimum_total=minimum_total,
        minimum_class=minimum_class,
    )
    return {
        "n": int(len(frame)),
        "positive_n": positive_n,
        "negative_n": negative_n,
        "support_status": support.status,
        "insufficient_reason": support.reason,
    }


def purge_validation_for_patient_disjoint(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    task_id: str = "",
    fold_index: int | str = "",
    fold_note: str = "",
    group_column: str = "public_patient_cluster_id",
    train_minimum_class: int = 10,
    validation_minimum_class: int = 5,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Purge validation rows that cannot be patient-disjoint from fold training.

    Training rows are returned unchanged. Validation rows are removed when the
    patient cluster is missing/unusable or is present in fold training. The
    returned audit row records before/after class support and both removal
    reasons.
    """

    train_present = _present_patient_cluster_mask(train, group_column)
    validation_present = _present_patient_cluster_mask(validation, group_column)
    if group_column in train and group_column in validation:
        train_groups = set(train.loc[train_present, group_column].astype(str))
        overlap = validation_present & validation[group_column].astype("string").isin(train_groups)
        missing = ~validation_present
    else:
        overlap = pd.Series(False, index=validation.index, dtype=bool)
        missing = pd.Series(True, index=validation.index, dtype=bool)
    keep = ~(overlap | missing)
    purged = validation.loc[keep].copy().reset_index(drop=True)
    train_support = _class_support_record(
        train,
        minimum_total=2 * train_minimum_class,
        minimum_class=train_minimum_class,
    )
    before_support = _class_support_record(
        validation,
        minimum_total=2 * validation_minimum_class,
        minimum_class=validation_minimum_class,
    )
    after_support = _class_support_record(
        purged,
        minimum_total=2 * validation_minimum_class,
        minimum_class=validation_minimum_class,
    )
    audit = {
        "task_id": task_id,
        "fold_index": fold_index,
        "fold_note": fold_note,
        "train_before_n": train_support["n"],
        "train_positive_n": train_support["positive_n"],
        "train_negative_n": train_support["negative_n"],
        "train_support_status": train_support["support_status"],
        "train_insufficient_reason": train_support["insufficient_reason"],
        "validation_before_n": before_support["n"],
        "validation_positive_before_n": before_support["positive_n"],
        "validation_negative_before_n": before_support["negative_n"],
        "removed_overlap_n": int(overlap.sum()),
        "removed_missing_patient_cluster_n": int(missing.sum()),
        "validation_after_n": after_support["n"],
        "validation_positive_after_n": after_support["positive_n"],
        "validation_negative_after_n": after_support["negative_n"],
        "validation_support_status": after_support["support_status"],
        "validation_insufficient_reason": after_support["insufficient_reason"],
        "purge_status": (
            "PASS"
            if train_support["support_status"] == "adequate"
            and after_support["support_status"] == "adequate"
            else "FAIL"
        ),
    }
    return purged, audit


def derive_patient_disjoint_test(
    development: pd.DataFrame,
    test: pd.DataFrame,
    *,
    task_id: str = "",
    group_column: str = "public_patient_cluster_id",
    minimum_total: int = 100,
    minimum_class: int = 20,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Derive a strict Site A test by removing overlap and unverifiable rows."""

    development_present = _present_patient_cluster_mask(development, group_column)
    test_present = _present_patient_cluster_mask(test, group_column)
    if group_column in development and group_column in test:
        development_groups = set(development.loc[development_present, group_column].astype(str))
        overlap = test_present & test[group_column].astype("string").isin(development_groups)
        missing = ~test_present
    else:
        overlap = pd.Series(False, index=test.index, dtype=bool)
        missing = pd.Series(True, index=test.index, dtype=bool)
    keep = ~(overlap | missing)
    strict_test = test.loc[keep].copy().reset_index(drop=True)
    before_support = _class_support_record(
        test,
        minimum_total=minimum_total,
        minimum_class=minimum_class,
    )
    after_support = _class_support_record(
        strict_test,
        minimum_total=minimum_total,
        minimum_class=minimum_class,
    )
    audit = {
        "task_id": task_id,
        "input_cohort_id": "site_a_test_all_samples",
        "output_cohort_id": "site_a_test_patient_disjoint",
        "n_before": before_support["n"],
        "positive_before_n": before_support["positive_n"],
        "negative_before_n": before_support["negative_n"],
        "removed_overlap_n": int(overlap.sum()),
        "removed_missing_patient_cluster_n": int(missing.sum()),
        "n_after": after_support["n"],
        "positive_after_n": after_support["positive_n"],
        "negative_after_n": after_support["negative_n"],
        "support_status": after_support["support_status"],
        "insufficient_reason": after_support["insufficient_reason"],
        "purge_status": "PASS" if after_support["support_status"] == "adequate" else "FAIL",
    }
    return strict_test, audit


def guard_platt_slope(slope: float) -> tuple[str, str]:
    """Return an explicit guard decision for a fitted Platt slope."""

    if not np.isfinite(slope):
        return "failed", "nonfinite_platt_slope"
    if slope <= 0:
        return "failed", "nonpositive_platt_slope"
    return "ok", ""


def fit_guarded_platt(y: np.ndarray, raw_probability: np.ndarray) -> GuardedPlattCalibration:
    """Fit source-OOF Platt calibration and reject ranking reversal."""

    y_array = np.asarray(y, dtype=np.int8)
    probability = np.asarray(raw_probability, dtype=float)
    if y_array.ndim != 1 or probability.shape != y_array.shape:
        raise ValueError("Platt labels and probabilities must be aligned one-dimensional arrays")
    if np.unique(y_array).size != 2:
        raise ValueError("Platt calibration requires both classes")
    model = fit_platt(y_array, probability)
    slope = float(model.coef_[0, 0])
    intercept = float(model.intercept_[0])
    status, reason = guard_platt_slope(slope)
    return GuardedPlattCalibration(
        model=model if status == "ok" else None,
        slope=slope,
        intercept=intercept,
        status=status,
        reason=reason,
    )


def select_fixed_thresholds(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Select all operating thresholds from source development OOF predictions."""

    return {
        "youden": youden_threshold(y, probability).threshold,
        "sensitivity90": threshold_at_min_sensitivity(y, probability, 0.90).threshold,
        "specificity90": threshold_at_min_specificity(y, probability, 0.90).threshold,
    }


def apply_fixed_thresholds(
    y: np.ndarray,
    probability: np.ndarray,
    thresholds: Mapping[str, float],
) -> dict[str, float]:
    """Apply preselected thresholds without re-estimating anything on test labels."""

    y_array = np.asarray(y, dtype=np.int8)
    p = np.asarray(probability, dtype=float)
    if y_array.shape != p.shape or y_array.ndim != 1:
        raise ValueError("Labels and probabilities must be aligned one-dimensional arrays")
    if not set(np.unique(y_array)).issubset({0, 1}):
        raise ValueError("Labels must be binary")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1]")
    output: dict[str, float] = {}
    for name, threshold_value in thresholds.items():
        threshold = float(threshold_value)
        if not np.isfinite(threshold):
            raise ValueError(f"Threshold {name} is not finite")
        predicted = p >= threshold
        positive = y_array == 1
        negative = ~positive
        tp = int(np.sum(predicted & positive))
        tn = int(np.sum(~predicted & negative))
        fp = int(np.sum(predicted & negative))
        fn = int(np.sum(~predicted & positive))
        sensitivity = tp / (tp + fn) if tp + fn else float("nan")
        specificity = tn / (tn + fp) if tn + fp else float("nan")
        ppv = tp / (tp + fp) if tp + fp else float("nan")
        npv = tn / (tn + fn) if tn + fn else float("nan")
        output.update(
            {
                f"threshold_{name}": threshold,
                f"sensitivity_{name}": float(sensitivity),
                f"specificity_{name}": float(specificity),
                f"ppv_{name}": float(ppv),
                f"npv_{name}": float(npv),
                f"false_susceptible_rate_{name}": float(1 - sensitivity),
                f"false_resistant_rate_{name}": float(1 - specificity),
            }
        )
    return output


def _resolve_feature_root(release_root: Path) -> Path:
    direct = release_root / "zd_mast_sample_metadata_public_v1.0.0.csv"
    if direct.is_file():
        return release_root
    candidates = sorted(
        path
        for path in release_root.glob("feature-release-*")
        if path.is_dir() and (path / direct.name).is_file()
    )
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one feature release, found {len(candidates)}")
    return candidates[0]


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def _matrix_binary_audit(
    matrix: np.ndarray,
    *,
    site_id: str,
    chunk_rows: int = 2048,
) -> dict[str, object]:
    nonbinary_n = 0
    nonfinite_n = 0
    for start in range(0, matrix.shape[0], chunk_rows):
        block = np.asarray(matrix[start : start + chunk_rows])
        finite = np.isfinite(block)
        nonfinite_n += int((~finite).sum())
        nonbinary_n += int((finite & (block != 0) & (block != 1)).sum())
    status = "PASS" if nonbinary_n == 0 and nonfinite_n == 0 else "FAIL"
    return {
        "gate": "binary_peak_presence_matrix",
        "site_id": site_id,
        "status": status,
        "n_rows": int(matrix.shape[0]),
        "n_features": int(matrix.shape[1]),
        "dtype": str(matrix.dtype),
        "nonbinary_value_n": nonbinary_n,
        "nonfinite_value_n": nonfinite_n,
    }


def _load_groups(feature_root: Path) -> tuple[pd.DataFrame, Path | None]:
    path = feature_root / "zd_mast_patient_episode_groups_public_v1.0.0.parquet"
    if not path.is_file():
        return pd.DataFrame(columns=["public_sample_id"]), None
    groups = pd.read_parquet(path)
    _require_columns(groups, ["public_sample_id"], "patient/episode groups")
    key = ["public_sample_id"] + (["task_id"] if "task_id" in groups else [])
    if groups.duplicated(key).any():
        raise ValueError(f"Duplicate patient/episode group rows on {key}")
    return groups, path


def load_analysis_inputs(
    release_root: Path,
    target_date_table: Path,
    *,
    task_ids: Sequence[str] = TASK_IDS,
    validate_binary_matrices: bool = True,
) -> AnalysisInputs:
    """Load and validate all release-level inputs without fitting a model."""

    selected_tasks = validate_task_ids(task_ids)
    root = release_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    feature_root = _resolve_feature_root(root)
    paths: dict[str, Path] = {
        "sample_metadata": feature_root / "zd_mast_sample_metadata_public_v1.0.0.csv",
        "historical_labels": feature_root / "zd_mast_ast_labels_historical_v1.0.0.parquet",
        "split_assignments": feature_root / "zd_mast_split_assignments_public_v1.0.0.csv",
        "rolling_origin_folds": feature_root
        / "zd_mast_protocol_b_rolling_origin_folds_public_v1.0.0.csv",
        "site_a_peak_presence6000": feature_root
        / "zd_mast_a_sample_level_peak_presence6000_v1.0.0.npy",
        "site_b_peak_presence6000": feature_root
        / "zd_mast_b_sample_level_peak_presence6000_v1.0.0.npy",
        "target_date_table": target_date_table.resolve(),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {role}: {path}")

    metadata = pd.read_csv(paths["sample_metadata"])
    _require_columns(metadata, ["site_id", "public_sample_id", "feature_row"], "sample metadata")
    metadata = metadata[metadata["site_id"].isin([SITE_A, SITE_B])].copy()
    metadata["public_sample_id"] = metadata["public_sample_id"].astype(str)
    if metadata.duplicated(["site_id", "public_sample_id"]).any():
        raise ValueError("Duplicate site/sample rows in sample metadata")
    if metadata["public_sample_id"].duplicated().any():
        raise ValueError("public_sample_id must be globally unique across sites")

    matrices: dict[str, np.ndarray] = {}
    matrix_rows: list[dict[str, object]] = []
    for site_id, role in (
        (SITE_A, "site_a_peak_presence6000"),
        (SITE_B, "site_b_peak_presence6000"),
    ):
        matrix = np.load(paths[role], mmap_mode="r")
        if matrix.ndim != 2 or matrix.shape[1] != N_FEATURES:
            raise ValueError(f"{site_id} peak-presence shape is {matrix.shape}, expected (*, 6000)")
        site_metadata = metadata[metadata["site_id"].eq(site_id)].copy()
        feature_rows = pd.to_numeric(site_metadata["feature_row"], errors="raise").astype(np.int64)
        if feature_rows.duplicated().any():
            raise ValueError(f"Duplicate feature_row values for {site_id}")
        if len(site_metadata) != matrix.shape[0] or set(feature_rows) != set(range(matrix.shape[0])):
            raise ValueError(
                f"{site_id} metadata/matrix mismatch: metadata={len(site_metadata)} matrix={matrix.shape[0]}"
            )
        matrices[site_id] = matrix
        if validate_binary_matrices:
            matrix_rows.append(_matrix_binary_audit(matrix, site_id=site_id))
        else:
            matrix_rows.append(
                {
                    "gate": "binary_peak_presence_matrix",
                    "site_id": site_id,
                    "status": "NOT_CHECKED",
                    "n_rows": int(matrix.shape[0]),
                    "n_features": int(matrix.shape[1]),
                    "dtype": str(matrix.dtype),
                    "nonbinary_value_n": pd.NA,
                    "nonfinite_value_n": pd.NA,
                }
            )

    labels = pd.read_parquet(paths["historical_labels"])
    _require_columns(
        labels,
        ["site_id", "public_sample_id", "task_id", "binary_s_vs_ir"],
        "historical labels",
    )
    labels = labels[
        labels["site_id"].isin([SITE_A, SITE_B]) & labels["task_id"].isin(selected_tasks)
    ].copy()
    labels["public_sample_id"] = labels["public_sample_id"].astype(str)
    reject_duplicate_labels(labels)
    invalid_label = ~labels["binary_s_vs_ir"].isin([0, 1])
    if invalid_label.any():
        raise ValueError(f"Non-binary historical S-vs-I/R labels: {int(invalid_label.sum())}")
    labels["y"] = labels["binary_s_vs_ir"].astype(np.int8)
    for site_id in (SITE_A, SITE_B):
        label_ids = set(labels.loc[labels["site_id"].eq(site_id), "public_sample_id"])
        metadata_ids = set(metadata.loc[metadata["site_id"].eq(site_id), "public_sample_id"])
        unknown_label_ids = label_ids - metadata_ids
        if unknown_label_ids:
            raise ValueError(
                f"Labels reference unknown {site_id} samples: n={len(unknown_label_ids)}"
            )

    splits = pd.read_csv(paths["split_assignments"])
    _require_columns(
        splits,
        ["analysis_id", "protocol", "site_id", "task_id", "public_sample_id", "split"],
        "split assignments",
    )
    folds = pd.read_csv(paths["rolling_origin_folds"])
    _require_columns(
        folds,
        ["analysis_id", "protocol", "task_id", "fold_index", "public_sample_id", "split"],
        "rolling-origin folds",
    )
    groups, groups_path = _load_groups(feature_root)
    if groups_path is not None:
        paths["patient_episode_groups"] = groups_path

    target_dates, target_date_source = load_target_date_table(target_date_table)
    if "public_patient_cluster_id" in target_dates:
        target_groups = target_dates[
            ["public_sample_id", "public_patient_cluster_id"]
        ].copy()
        group_columns = list(groups.columns)
        for column in group_columns:
            if column not in target_groups:
                target_groups[column] = pd.NA
        for column in target_groups.columns:
            if column not in groups:
                groups[column] = pd.NA
                group_columns.append(column)
        groups = pd.concat(
            [groups[group_columns], target_groups[group_columns]], ignore_index=True
        )
        group_key = ["public_sample_id"] + (["task_id"] if "task_id" in groups else [])
        if groups.duplicated(group_key).any():
            raise ValueError("Duplicate patient grouping rows after adding Site B groups")
    target_ids = set(target_dates["public_sample_id"])
    site_b_metadata_ids = set(metadata.loc[metadata["site_id"].eq(SITE_B), "public_sample_id"])
    unknown_date_ids = target_ids - site_b_metadata_ids
    if unknown_date_ids:
        raise ValueError(f"Target-date table references non-Site-B samples: n={len(unknown_date_ids)}")

    audit = pd.DataFrame(matrix_rows)
    if validate_binary_matrices and not audit["status"].eq("PASS").all():
        raise ValueError("One or more peak_presence6000 matrices contain invalid values")
    return AnalysisInputs(
        feature_root=feature_root,
        matrices=matrices,
        sample_metadata=metadata,
        labels=labels,
        groups=groups,
        splits=splits,
        folds=folds,
        target_dates=target_dates,
        target_date_source_column=target_date_source,
        input_paths=paths,
        matrix_audit=audit,
    )


def _attach_groups(frame: pd.DataFrame, groups: pd.DataFrame, task_id: str) -> pd.DataFrame:
    output = frame.copy()
    if groups.empty:
        output["public_patient_cluster_id"] = pd.NA
        output["public_episode_id"] = pd.NA
        return output
    selected = groups.copy()
    if "task_id" in selected:
        task_values = selected["task_id"].astype("string")
        selected = selected[
            task_values.eq(task_id) | task_values.isna() | task_values.str.strip().eq("")
        ].drop(columns="task_id")
    if selected["public_sample_id"].duplicated().any():
        raise ValueError(f"Duplicate patient grouping rows for {task_id}")
    keep = [
        column
        for column in (
            "public_sample_id",
            "public_patient_cluster_id",
            "public_episode_id",
            "episode_first_sample_flag",
        )
        if column in selected
    ]
    output = output.merge(selected[keep], on="public_sample_id", how="left", validate="one_to_one")
    if "public_patient_cluster_id" not in output:
        output["public_patient_cluster_id"] = pd.NA
    if "public_episode_id" not in output:
        output["public_episode_id"] = pd.NA
    return output


def _task_base(inputs: AnalysisInputs, task_id: str, site_id: str) -> pd.DataFrame:
    labels = inputs.labels[
        inputs.labels["task_id"].eq(task_id) & inputs.labels["site_id"].eq(site_id)
    ][["public_sample_id", "y"]].copy()
    metadata = inputs.sample_metadata[inputs.sample_metadata["site_id"].eq(site_id)][
        ["public_sample_id", "feature_row"]
    ].copy()
    base = labels.merge(metadata, on="public_sample_id", how="inner", validate="one_to_one")
    if len(base) != len(labels):
        raise ValueError(f"{site_id} {task_id}: labels without feature rows")
    return _attach_groups(base, inputs.groups, task_id)


def build_task_cohorts(inputs: AnalysisInputs, task_id: str) -> TaskCohorts:
    """Build frozen Site A and date-filtered Site B cohorts for one task."""

    validate_task_ids([task_id])
    source_base = _task_base(inputs, task_id, SITE_A)
    source_split = inputs.splits[
        inputs.splits["analysis_id"].eq("local_temporal")
        & inputs.splits["protocol"].eq("current_workflow_protocol_b")
        & inputs.splits["site_id"].eq(SITE_A)
        & inputs.splits["task_id"].eq(task_id)
        & inputs.splits["split"].isin(["train", "test"])
    ].copy()
    source_split["public_sample_id"] = source_split["public_sample_id"].astype(str)
    if source_split.duplicated(["public_sample_id", "split"]).any():
        raise ValueError(f"{task_id}: duplicate source split assignment")
    assigned = source_split.merge(source_base, on="public_sample_id", how="inner", validate="one_to_one")
    if len(assigned) != len(source_split):
        raise ValueError(f"{task_id}: source split references unusable samples")
    sort_columns = [column for column in ("row_order", "public_sample_id") if column in assigned]
    development = assigned[assigned["split"].eq("train")].sort_values(sort_columns).reset_index(drop=True)
    source_test_all_samples = (
        assigned[assigned["split"].eq("test")]
        .sort_values(sort_columns)
        .reset_index(drop=True)
    )
    reject_source_overlap(development, source_test_all_samples)
    source_test_patient_disjoint, source_test_audit = derive_patient_disjoint_test(
        development,
        source_test_all_samples,
        task_id=task_id,
    )

    development_ids = set(development["public_sample_id"].astype(str))
    test_ids = set(source_test_all_samples["public_sample_id"].astype(str))
    fold_table = inputs.folds[
        inputs.folds["analysis_id"].eq("local_temporal")
        & inputs.folds["protocol"].eq("current_workflow_protocol_b")
        & inputs.folds["task_id"].eq(task_id)
    ].copy()
    fold_table["public_sample_id"] = fold_table["public_sample_id"].astype(str)
    if fold_table.duplicated(["fold_index", "public_sample_id", "split"]).any():
        raise ValueError(f"{task_id}: duplicate rolling-origin assignment")
    if not set(fold_table["public_sample_id"].astype(str)).issubset(development_ids):
        raise ValueError(f"{task_id}: rolling-origin folds include non-development samples")
    if set(fold_table["public_sample_id"].astype(str)) & test_ids:
        raise ValueError(f"{task_id}: source test sample leaked into rolling-origin folds")
    source_folds: list[tuple[pd.DataFrame, pd.DataFrame, str]] = []
    fold_audit_rows: list[dict[str, object]] = []
    validation_ids_seen: set[str] = set()
    for fold_index in sorted(fold_table["fold_index"].dropna().unique()):
        one = fold_table[fold_table["fold_index"].eq(fold_index)].copy()
        if not set(one["split"].astype(str)).issubset({"train", "validation"}):
            raise ValueError(f"{task_id}: rolling-origin fold has unsupported split values")
        one = one.merge(source_base, on="public_sample_id", how="inner", validate="many_to_one")
        train = one[one["split"].eq("train")].sort_values(
            [column for column in ("row_order", "public_sample_id") if column in one]
        ).reset_index(drop=True)
        validation = one[one["split"].eq("validation")].sort_values(
            [column for column in ("row_order", "public_sample_id") if column in one]
        ).reset_index(drop=True)
        reject_source_overlap(train, validation)
        note_values = one.get("fold_note", pd.Series(dtype="string")).dropna().unique()
        note = str(note_values[0]) if len(note_values) else f"fold={fold_index}"
        validation, fold_audit = purge_validation_for_patient_disjoint(
            train,
            validation,
            task_id=task_id,
            fold_index=fold_index,
            fold_note=note,
        )
        fold_audit_rows.append(fold_audit)
        fold_validation_ids = set(validation["public_sample_id"].astype(str))
        repeated_validation_ids = validation_ids_seen & fold_validation_ids
        if repeated_validation_ids:
            raise ValueError(
                f"{task_id}: rolling-origin validation sample repeated across folds: "
                f"n={len(repeated_validation_ids)}"
            )
        validation_ids_seen.update(fold_validation_ids)
        source_folds.append((train, validation, note))

    target_base = _task_base(inputs, task_id, SITE_B)
    target_base = attach_target_dates(target_base, inputs.target_dates)
    target_dated = target_base[target_base["target_date"].notna()].copy()
    primary_start, primary_end, _ = COHORT_WINDOWS["site_b_primary"]
    full_start, full_end, _ = COHORT_WINDOWS["site_b_full_period_sensitivity"]
    target_primary = filter_date_window(target_dated, primary_start, primary_end)
    target_full = filter_date_window(target_dated, full_start, full_end)
    return TaskCohorts(
        task_id=task_id,
        source_development=development,
        source_test_patient_disjoint=source_test_patient_disjoint,
        source_test_all_samples=source_test_all_samples,
        source_folds=source_folds,
        source_test_purge_audit=pd.DataFrame([source_test_audit]),
        fold_purge_audit=pd.DataFrame(fold_audit_rows),
        target_primary=target_primary,
        target_full_period=target_full,
    )


def _cohort_count_row(
    task_id: str,
    cohort_id: str,
    frame: pd.DataFrame,
    *,
    site_id: str,
    date_start: str,
    date_end: str,
    minimum_total: int = 100,
    minimum_class: int = 20,
) -> dict[str, object]:
    positive_n = int(frame["y"].sum()) if len(frame) else 0
    negative_n = int(frame["y"].eq(0).sum()) if len(frame) else 0
    support = classify_support(
        len(frame), positive_n, negative_n, minimum_total=minimum_total, minimum_class=minimum_class
    )
    patient = frame.get("public_patient_cluster_id", pd.Series(pd.NA, index=frame.index))
    patient_present = _present_patient_cluster_mask(frame, "public_patient_cluster_id")
    return {
        "task_id": task_id,
        "cohort_id": cohort_id,
        "site_id": site_id,
        "date_start": date_start,
        "date_end": date_end,
        "n_rows": int(len(frame)),
        "unique_sample_n": int(frame["public_sample_id"].nunique()),
        "positive_n": positive_n,
        "negative_n": negative_n,
        "positive_rate": float(positive_n / len(frame)) if len(frame) else float("nan"),
        "patient_cluster_n": int(patient.loc[patient_present].astype(str).nunique()),
        "missing_patient_cluster_n": int((~patient_present).sum()),
        "support_status": support.status,
        "insufficient_reason": support.reason,
        "target_labels_used_for_selection": False,
    }


def preflight_tables(
    inputs: AnalysisInputs,
    task_ids: Sequence[str] = TASK_IDS,
) -> tuple[dict[str, TaskCohorts], pd.DataFrame, pd.DataFrame]:
    """Build cohort counts and auditable validation findings without training."""

    tasks = validate_task_ids(task_ids)
    cohorts: dict[str, TaskCohorts] = {}
    count_rows: list[dict[str, object]] = []
    findings: list[dict[str, object]] = []
    for row in inputs.matrix_audit.to_dict("records"):
        findings.append(
            {
                "gate": row["gate"],
                "task_id": "ALL",
                "cohort_id": row["site_id"],
                "status": row["status"],
                "detail": json.dumps(row, sort_keys=True, default=str),
            }
        )

    site_b_label_ids = set(
        inputs.labels.loc[inputs.labels["site_id"].eq(SITE_B), "public_sample_id"].astype(str)
    )
    dated_ids = set(inputs.target_dates["public_sample_id"].astype(str))
    missing_date_n = len(site_b_label_ids - dated_ids)
    findings.append(
        {
            "gate": "site_b_target_date_coverage",
            "task_id": "ALL",
            "cohort_id": SITE_B,
            "status": "PASS" if missing_date_n == 0 else "WARN",
            "detail": f"labelled_samples_without_target_date={missing_date_n}",
        }
    )

    for task_id in tasks:
        task = build_task_cohorts(inputs, task_id)
        cohorts[task_id] = task
        count_rows.extend(
            [
                _cohort_count_row(
                    task_id,
                    "site_a_development",
                    task.source_development,
                    site_id=SITE_A,
                    date_start="2025-07-01",
                    date_end="2026-02-28",
                ),
                _cohort_count_row(
                    task_id,
                    "site_a_test_patient_disjoint",
                    task.source_test_patient_disjoint,
                    site_id=SITE_A,
                    date_start="2026-03-01",
                    date_end="2026-06-09",
                ),
                _cohort_count_row(
                    task_id,
                    "site_a_test_all_samples",
                    task.source_test_all_samples,
                    site_id=SITE_A,
                    date_start="2026-03-01",
                    date_end="2026-06-09",
                ),
                _cohort_count_row(
                    task_id,
                    "site_b_primary",
                    task.target_primary,
                    site_id=SITE_B,
                    date_start="2026-03-01",
                    date_end="2026-06-09",
                ),
                _cohort_count_row(
                    task_id,
                    "site_b_full_period_sensitivity",
                    task.target_full_period,
                    site_id=SITE_B,
                    date_start="2026-01-01",
                    date_end="2026-07-29",
                ),
            ]
        )
        fold_problem: list[str] = []
        if len(task.source_folds) < 2:
            fold_problem.append("fewer_than_two_folds")
        for audit in task.fold_purge_audit.to_dict("records"):
            fold_status = str(audit["purge_status"])
            if fold_status != "PASS":
                fold_problem.append(
                    f"fold_{audit['fold_index']}:"
                    f"train={audit['train_support_status']};"
                    f"validation={audit['validation_support_status']}"
                )
            findings.append(
                {
                    "gate": "source_rolling_origin_fold_patient_disjoint",
                    "task_id": task_id,
                    "cohort_id": f"fold_{audit['fold_index']}",
                    "status": fold_status,
                    "detail": json.dumps(audit, sort_keys=True, default=str),
                }
            )
        findings.append(
            {
                "gate": "source_rolling_origin_support",
                "task_id": task_id,
                "cohort_id": "site_a_development",
                "status": "PASS" if not fold_problem else "FAIL",
                "detail": ";".join(fold_problem),
            }
        )
        source_test_audit = task.source_test_purge_audit.iloc[0].to_dict()
        findings.append(
            {
                "gate": "source_patient_disjoint_test_purge",
                "task_id": task_id,
                "cohort_id": "site_a_test_patient_disjoint",
                "status": "PASS",
                "detail": json.dumps(source_test_audit, sort_keys=True, default=str),
            }
        )
        findings.append(
            {
                "gate": "source_patient_disjoint_test_support",
                "task_id": task_id,
                "cohort_id": "site_a_test_patient_disjoint",
                "status": (
                    "PASS"
                    if source_test_audit["support_status"] == "adequate"
                    else "FAIL"
                ),
                "detail": json.dumps(source_test_audit, sort_keys=True, default=str),
            }
        )
    counts = pd.DataFrame(count_rows)
    for row in counts.to_dict("records"):
        findings.append(
            {
                "gate": "cohort_support",
                "task_id": row["task_id"],
                "cohort_id": row["cohort_id"],
                "status": "PASS" if row["support_status"] == "adequate" else "WARN",
                "detail": row["insufficient_reason"],
            }
        )
    return cohorts, counts, pd.DataFrame(findings)


def _bootstrap_groups(frame: pd.DataFrame) -> tuple[np.ndarray, str]:
    patient = frame.get("public_patient_cluster_id", pd.Series(pd.NA, index=frame.index))
    patient = patient.astype("string")
    patient_present = _present_patient_cluster_mask(frame, "public_patient_cluster_id")
    patient = patient.where(patient_present)
    fallback = "sample:" + frame["public_sample_id"].astype(str)
    source = "public_patient_cluster_id_with_sample_fallback"
    return patient.fillna(fallback).to_numpy(dtype=str), source


def _resample_cluster_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(groups, dtype=str)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Bootstrap groups must be a non-empty one-dimensional array")
    unique = pd.unique(values)
    index_by_group = {group: np.flatnonzero(values == group) for group in unique}
    selected = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([index_by_group[group] for group in selected])


def _metric_function(name: str) -> Callable[[np.ndarray, np.ndarray], float]:
    if name == "auroc":
        return lambda y, p: float(roc_auc_score(y, p))
    if name == "auprc":
        return lambda y, p: float(average_precision_score(y, p))
    if name == "brier":
        return lambda y, p: float(brier_score_loss(y, p))
    if name == "ece":
        return lambda y, p: float(expected_calibration_error(y, p))
    raise ValueError(f"Unsupported bootstrap metric: {name}")


def independent_cluster_bootstrap_delta(
    site_a_y: np.ndarray,
    site_a_probability: np.ndarray,
    site_a_groups: np.ndarray,
    site_b_y: np.ndarray,
    site_b_probability: np.ndarray,
    site_b_groups: np.ndarray,
    *,
    metric: str,
    n_boot: int = DEFAULT_BOOTSTRAP_COUNT,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Independently resample site clusters and return target-minus-source draws."""

    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    a_y = np.asarray(site_a_y, dtype=np.int8)
    a_p = np.asarray(site_a_probability, dtype=float)
    b_y = np.asarray(site_b_y, dtype=np.int8)
    b_p = np.asarray(site_b_probability, dtype=float)
    a_groups = np.asarray(site_a_groups, dtype=str)
    b_groups = np.asarray(site_b_groups, dtype=str)
    if not (a_y.shape == a_p.shape == a_groups.shape):
        raise ValueError("Site A bootstrap arrays are misaligned")
    if not (b_y.shape == b_p.shape == b_groups.shape):
        raise ValueError("Site B bootstrap arrays are misaligned")
    if not np.isfinite(a_p).all() or not np.isfinite(b_p).all():
        raise ValueError("Bootstrap probabilities must be finite")
    if ((a_p < 0) | (a_p > 1)).any() or ((b_p < 0) | (b_p > 1)).any():
        raise ValueError("Bootstrap probabilities must be in [0, 1]")
    function = _metric_function(metric)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for replicate in range(n_boot):
        a_index = _resample_cluster_indices(a_groups, rng)
        b_index = _resample_cluster_indices(b_groups, rng)
        if metric in {"auroc", "auprc"} and (
            np.unique(a_y[a_index]).size < 2 or np.unique(b_y[b_index]).size < 2
        ):
            a_value = b_value = delta = float("nan")
        else:
            a_value = function(a_y[a_index], a_p[a_index])
            b_value = function(b_y[b_index], b_p[b_index])
            delta = b_value - a_value
        rows.append(
            {
                "bootstrap_replicate": replicate,
                "site_a_metric": a_value,
                "site_b_metric": b_value,
                "delta_site_b_minus_site_a": delta,
            }
        )
    return pd.DataFrame(rows)


def _safe_calibration(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    if np.unique(y).size < 2:
        return float("nan"), float("nan")
    try:
        return calibration_slope_intercept(y, probability)
    except (ValueError, FloatingPointError):
        return float("nan"), float("nan")


def probability_metrics(
    y: np.ndarray,
    raw_probability: np.ndarray,
    calibrated_probability: np.ndarray | None,
    thresholds: Mapping[str, float],
) -> dict[str, float | bool]:
    """Return explicit raw, calibrated, calibration, and operating metrics."""

    y_array = np.asarray(y, dtype=np.int8)
    raw = np.asarray(raw_probability, dtype=float)
    if y_array.shape != raw.shape or y_array.ndim != 1:
        raise ValueError("Evaluation labels and raw probabilities are misaligned")
    baseline = float(y_array.mean()) if len(y_array) else float("nan")
    has_two_classes = np.unique(y_array).size == 2
    raw_auprc = float(average_precision_score(y_array, raw)) if has_two_classes else float("nan")
    raw_slope, raw_intercept = _safe_calibration(y_array, raw)
    result: dict[str, float | bool] = {
        "raw_auroc": float(roc_auc_score(y_array, raw)) if has_two_classes else float("nan"),
        "raw_auprc": raw_auprc,
        "auprc_baseline": baseline,
        "raw_auprc_lift": raw_auprc / baseline if baseline > 0 else float("nan"),
        "raw_brier": float(brier_score_loss(y_array, raw)) if len(y_array) else float("nan"),
        "raw_ece": expected_calibration_error(y_array, raw) if len(y_array) else float("nan"),
        "raw_calibration_slope": raw_slope,
        "raw_calibration_intercept": raw_intercept,
        "threshold_metrics_valid": False,
    }
    calibrated_names = (
        "calibrated_auroc",
        "calibrated_auprc",
        "calibrated_auprc_lift",
        "calibrated_brier",
        "calibrated_ece",
        "calibrated_calibration_slope",
        "calibrated_calibration_intercept",
    )
    for threshold_name in THRESHOLD_NAMES:
        for metric_name in THRESHOLD_METRIC_NAMES:
            result[f"{metric_name}_{threshold_name}"] = float("nan")

    if calibrated_probability is None:
        result.update({name: float("nan") for name in calibrated_names})
        return result

    calibrated = np.asarray(calibrated_probability, dtype=float)
    if calibrated.shape != y_array.shape:
        raise ValueError("Calibrated probabilities are misaligned")
    if not np.isfinite(calibrated).all() or ((calibrated < 0) | (calibrated > 1)).any():
        raise ValueError("Calibrated probabilities must be finite values in [0, 1]")
    calibrated_auprc = (
        float(average_precision_score(y_array, calibrated)) if has_two_classes else float("nan")
    )
    calibrated_slope, calibrated_intercept = _safe_calibration(y_array, calibrated)
    result.update(
        {
            "calibrated_auroc": (
                float(roc_auc_score(y_array, calibrated)) if has_two_classes else float("nan")
            ),
            "calibrated_auprc": calibrated_auprc,
            "calibrated_auprc_lift": (
                calibrated_auprc / baseline if baseline > 0 else float("nan")
            ),
            "calibrated_brier": float(brier_score_loss(y_array, calibrated)),
            "calibrated_ece": expected_calibration_error(y_array, calibrated),
            "calibrated_calibration_slope": calibrated_slope,
            "calibrated_calibration_intercept": calibrated_intercept,
            "threshold_metrics_valid": bool(has_two_classes and thresholds),
        }
    )
    if has_two_classes and thresholds:
        result.update(apply_fixed_thresholds(y_array, calibrated, thresholds))
    return result


def _cluster_bootstrap_intervals(
    y: np.ndarray,
    raw: np.ndarray,
    calibrated: np.ndarray | None,
    groups: np.ndarray,
    thresholds: Mapping[str, float],
    *,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if not (len(y) == len(raw) == len(groups)):
        raise ValueError("Bootstrap arrays are misaligned")
    rng = np.random.default_rng(seed)
    collected: dict[str, list[float]] = {}
    for _ in range(n_boot):
        index = _resample_cluster_indices(groups, rng)
        boot_y = y[index]
        if np.unique(boot_y).size < 2:
            continue
        boot_calibrated = calibrated[index] if calibrated is not None else None
        values = probability_metrics(boot_y, raw[index], boot_calibrated, thresholds)
        for name, value in values.items():
            if name.startswith("threshold_") or name in {
                "auprc_baseline",
                "threshold_metrics_valid",
                "raw_calibration_slope",
                "raw_calibration_intercept",
                "calibrated_calibration_slope",
                "calibrated_calibration_intercept",
            }:
                continue
            numeric = float(value)
            if np.isfinite(numeric):
                collected.setdefault(name, []).append(numeric)
    rows: list[dict[str, object]] = []
    for metric_name, values in sorted(collected.items()):
        array = np.asarray(values, dtype=float)
        rows.append(
            {
                "metric": metric_name,
                "bootstrap_requested_n": n_boot,
                "bootstrap_valid_n": int(len(array)),
                "ci_low": float(np.quantile(array, 0.025)),
                "ci_high": float(np.quantile(array, 0.975)),
                "bootstrap_median": float(np.median(array)),
            }
        )
    return pd.DataFrame(rows)


def fit_source_task(
    matrix: np.ndarray,
    cohorts: TaskCohorts,
    *,
    threads: int,
    seed: int,
) -> FittedSourceTask:
    """Tune, calibrate, threshold, and fit using Site A development only."""

    source = PublicProtocolData(
        task_id=cohorts.task_id,
        feature_matrix=matrix,
        development=cohorts.source_development,
        test=cohorts.source_test_patient_disjoint,
        folds=cohorts.source_folds,
    )
    development_ids = set(source.development["public_sample_id"].astype(str))
    test_ids = set(source.test["public_sample_id"].astype(str))
    if development_ids & test_ids:
        raise ValueError(f"{cohorts.task_id}: source development/test overlap")
    for train, validation, _ in source.folds:
        fold_ids = set(train["public_sample_id"].astype(str)) | set(
            validation["public_sample_id"].astype(str)
        )
        if not fold_ids.issubset(development_ids) or fold_ids & test_ids:
            raise ValueError(f"{cohorts.task_id}: non-development sample in OOF folds")
        reject_patient_cluster_overlap(train, validation)
    reject_patient_cluster_overlap(source.development, source.test)

    params, tuning = tune_with_frozen_folds(source, seed, threads)
    oof = out_of_fold_predictions(source, params, seed + 10_000, threads)
    oof_y = oof["y"].to_numpy(dtype=np.int8)
    calibrator = fit_guarded_platt(oof_y, oof["raw_probability"].to_numpy(dtype=float))
    thresholds: dict[str, float] = {}
    calibrated_oof = calibrator.apply(oof["raw_probability"].to_numpy(dtype=float))
    if calibrated_oof is not None:
        oof["calibrated_probability"] = calibrated_oof
        thresholds = select_fixed_thresholds(oof_y, calibrated_oof)
    else:
        oof["calibrated_probability"] = np.nan

    x_development, y_development = matrix_rows(matrix, source.development)
    model = fit_lightgbm(params, x_development, y_development, seed + 20_000, threads)
    tuning = tuning.copy()
    tuning.insert(0, "task_id", cohorts.task_id)
    return FittedSourceTask(
        task_id=cohorts.task_id,
        model=model,
        calibrator=calibrator,
        thresholds=thresholds,
        hyperparameters=params,
        tuning=tuning,
        oof=oof,
        seed=seed,
    )


def evaluate_fitted_task(
    fitted: FittedSourceTask,
    matrix: np.ndarray,
    frame: pd.DataFrame,
    *,
    cohort_id: str,
    site_id: str,
    date_start: str,
    date_end: str,
    n_boot: int,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Evaluate an unchanged fitted source model on one frozen cohort."""

    x, y = matrix_rows(matrix, frame)
    raw = predict(fitted.model, x)
    calibrated = fitted.calibrator.apply(raw)
    positive_n = int(y.sum())
    negative_n = int(y.size - positive_n)
    support = classify_support(len(y), positive_n, negative_n)
    groups, group_source = _bootstrap_groups(frame)
    metrics = probability_metrics(y, raw, calibrated, fitted.thresholds)
    row: dict[str, object] = {
        "analysis_id": "cross_platform_like_for_like",
        "task_id": fitted.task_id,
        "model": "lightgbm",
        "endpoint": "historical_S_vs_IR",
        "feature_representation": "peak_presence6000",
        "cohort_id": cohort_id,
        "site_id": site_id,
        "date_start": date_start,
        "date_end": date_end,
        "n_test": int(len(y)),
        "positive_n": positive_n,
        "negative_n": negative_n,
        "positive_rate": float(y.mean()) if len(y) else float("nan"),
        "patient_cluster_n": int(pd.Series(groups).nunique()),
        "bootstrap_group_source": group_source,
        "support_status": support.status,
        "insufficient_reason": support.reason,
        "calibration_status": fitted.calibrator.status,
        "calibration_failure_reason": fitted.calibrator.reason,
        "source_oof_platt_slope": fitted.calibrator.slope,
        "source_oof_platt_intercept": fitted.calibrator.intercept,
        "same_fitted_source_model": True,
        "target_labels_used_for_training_or_selection": False,
        "best_hyperparameters": json.dumps(fitted.hyperparameters, sort_keys=True),
        **metrics,
    }
    bootstrap = pd.DataFrame()
    if support.eligible_for_discrimination and n_boot > 0:
        bootstrap = _cluster_bootstrap_intervals(
            y,
            raw,
            calibrated,
            groups,
            fitted.thresholds,
            n_boot=n_boot,
            seed=seed,
        )
        if not bootstrap.empty:
            bootstrap.insert(0, "site_id", site_id)
            bootstrap.insert(0, "cohort_id", cohort_id)
            bootstrap.insert(0, "task_id", fitted.task_id)
            for interval in bootstrap.itertuples(index=False):
                row[f"{interval.metric}_ci_low"] = interval.ci_low
                row[f"{interval.metric}_ci_high"] = interval.ci_high
    predictions = frame[["public_sample_id", "public_patient_cluster_id", "y"]].copy()
    predictions.insert(0, "site_id", site_id)
    predictions.insert(0, "cohort_id", cohort_id)
    predictions.insert(0, "task_id", fitted.task_id)
    predictions["raw_probability"] = raw
    predictions["calibrated_probability"] = calibrated if calibrated is not None else np.nan
    return row, bootstrap, predictions


def site_difference_intervals(
    task_id: str,
    source_predictions: pd.DataFrame,
    target_predictions: pd.DataFrame,
    *,
    source_cohort_id: str = "site_a_test_patient_disjoint",
    comparison_role: str = "primary",
    target_cohort_id: str,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Calculate independent cluster-bootstrap target-minus-source differences."""

    source_groups, _ = _bootstrap_groups(source_predictions)
    target_groups, _ = _bootstrap_groups(target_predictions)
    rows: list[dict[str, object]] = []
    probability_specs = [("raw", "raw_probability")]
    if (
        source_predictions["calibrated_probability"].notna().all()
        and target_predictions["calibrated_probability"].notna().all()
    ):
        probability_specs.append(("calibrated", "calibrated_probability"))
    for probability_type, column in probability_specs:
        for metric in ("auroc", "auprc"):
            draws = independent_cluster_bootstrap_delta(
                source_predictions["y"].to_numpy(dtype=np.int8),
                source_predictions[column].to_numpy(dtype=float),
                source_groups,
                target_predictions["y"].to_numpy(dtype=np.int8),
                target_predictions[column].to_numpy(dtype=float),
                target_groups,
                metric=metric,
                n_boot=n_boot,
                seed=deterministic_seed(seed, task_id, target_cohort_id, probability_type, metric),
            )
            valid = draws["delta_site_b_minus_site_a"].dropna().to_numpy(dtype=float)
            source_point = _metric_function(metric)(
                source_predictions["y"].to_numpy(dtype=np.int8),
                source_predictions[column].to_numpy(dtype=float),
            )
            target_point = _metric_function(metric)(
                target_predictions["y"].to_numpy(dtype=np.int8),
                target_predictions[column].to_numpy(dtype=float),
            )
            rows.append(
                {
                    "task_id": task_id,
                    "source_cohort_id": source_cohort_id,
                    "target_cohort_id": target_cohort_id,
                    "comparison_role": comparison_role,
                    "probability_type": probability_type,
                    "metric": metric,
                    "source_estimate": source_point,
                    "target_estimate": target_point,
                    "delta_site_b_minus_site_a": target_point - source_point,
                    "bootstrap_method": "independent_site_wise_patient_cluster",
                    "bootstrap_requested_n": n_boot,
                    "bootstrap_valid_n": int(len(valid)),
                    "ci_low": float(np.quantile(valid, 0.025)) if len(valid) else float("nan"),
                    "ci_high": float(np.quantile(valid, 0.975)) if len(valid) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def manageable_file_record(
    path: Path,
    *,
    role: str,
    feature_root: Path,
    max_hash_bytes: int = MAX_METADATA_HASH_BYTES,
) -> dict[str, object]:
    """Build a privacy-conscious input record with SHA256 when manageable."""

    size = path.stat().st_size
    try:
        display_path = path.relative_to(feature_root).as_posix()
    except ValueError:
        display_path = path.name
    digest: str | None = None
    hash_status = "skipped_size"
    if size <= max_hash_bytes:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                hasher.update(block)
        digest = hasher.hexdigest()
        hash_status = "computed"
    return {
        "role": role,
        "path": display_path,
        "size_bytes": size,
        "sha256": digest,
        "hash_status": hash_status,
    }
