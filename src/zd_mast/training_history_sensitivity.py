"""Deterministic size-matched training-history sensitivity analysis.

This module isolates reviewer concern R2 from the original same-test analysis.
For each task, the pre-marker and current-workflow source pools define a common
per-class cap. Every training regime then receives the same negative and
positive sample counts at each learning fraction. The pooled regime splits each
class as evenly as possible between eras. Sampling orders are deterministic and
nested within a repeat, while the future patient-disjoint test cohort and the
task-specific frozen LightGBM parameters remain unchanged.

Only raw discrimination metrics are calculated. Test labels are never used for
sampling, hyperparameter selection, calibration, or threshold selection.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .cross_platform import deterministic_seed
from .modeling import fit_lightgbm, matrix_rows, predict
from .training_history import HistoryTaskCohorts, TrainingHistoryInputs


ANALYSIS_ID = "training_history_size_matched_v2"
DEFAULT_SEED = 20260817
REQUIRED_LEARNING_FRACTIONS: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)

CURRENT_ONLY = "current_only"
PRE_MARKER_ONLY = "pre_marker_only"
POOLED_ERA_BALANCED = "pooled_era_balanced"
TRAINING_REGIMES: tuple[str, ...] = (
    CURRENT_ONLY,
    PRE_MARKER_ONLY,
    POOLED_ERA_BALANCED,
)

PRE_ERA = "pre_marker"
CURRENT_ERA = "current_workflow"

PAIRED_COMPARISONS: tuple[tuple[str, str], ...] = (
    (PRE_MARKER_ONLY, CURRENT_ONLY),
    (POOLED_ERA_BALANCED, CURRENT_ONLY),
    (POOLED_ERA_BALANCED, PRE_MARKER_ONLY),
)

PAIRED_DELTA_COLUMNS: tuple[str, ...] = (
    "analysis_id",
    "task_id",
    "repeat_index",
    "learning_fraction",
    "comparison_id",
    "comparator_regime",
    "reference_regime",
    "metric",
    "comparator_value",
    "reference_value",
    "delta_comparator_minus_reference",
    "n_development_per_regime",
    "development_positive_n_per_regime",
    "development_negative_n_per_regime",
    "n_test",
    "fixed_test_signature",
)

DELTA_SUMMARY_COLUMNS: tuple[str, ...] = (
    "task_id",
    "learning_fraction",
    "comparison_id",
    "comparator_regime",
    "reference_regime",
    "metric",
    "repeat_n",
    "delta_mean",
    "delta_median",
    "delta_sd",
    "repeat_distribution_q025",
    "repeat_distribution_q975",
    "fraction_delta_gt_zero",
)

REQUIRED_POOL_COLUMNS: frozenset[str] = frozenset(
    {
        "public_sample_id",
        "public_patient_cluster_id",
        "feature_row",
        "row_order",
        "y",
    }
)


@dataclass(frozen=True)
class SampledTrainingCell:
    """One deterministic task/repeat/fraction/regime training cohort."""

    task_id: str
    repeat_index: int
    learning_fraction: float
    training_regime: str
    frame: pd.DataFrame
    target_negative_n: int
    target_positive_n: int
    pre_era_n: int
    current_era_n: int
    status: str
    insufficient_reason: str
    sample_signature: str


def validate_learning_fractions(
    fractions: Sequence[float],
    *,
    require_prespecified_grid: bool = True,
) -> tuple[float, ...]:
    """Validate and canonicalize learning fractions.

    The full CLI analysis requires the prespecified 0.25/0.50/0.75/1.00 grid.
    Tests and reusable helpers may explicitly disable that requirement.
    """

    values = tuple(sorted(float(value) for value in fractions))
    if not values:
        raise ValueError("At least one learning fraction is required")
    if any(not np.isfinite(value) or value <= 0 or value > 1 for value in values):
        raise ValueError("Learning fractions must be finite values in (0, 1]")
    if len(values) != len(set(values)):
        raise ValueError("Learning fractions must be unique")
    if require_prespecified_grid and (
        len(values) != len(REQUIRED_LEARNING_FRACTIONS)
        or any(
            not np.isclose(actual, expected, rtol=0, atol=1e-12)
            for actual, expected in zip(values, REQUIRED_LEARNING_FRACTIONS, strict=True)
        )
    ):
        raise ValueError(
            "The full analysis requires learning fractions 0.25, 0.50, 0.75, and 1.00"
        )
    return values


def _missing_text(series: pd.Series) -> pd.Series:
    text = series.astype("string")
    return text.isna() | text.str.strip().eq("")


def _canonical_pool(frame: pd.DataFrame, *, era: str, task_id: str) -> pd.DataFrame:
    missing = REQUIRED_POOL_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{task_id} {era}: missing columns {sorted(missing)}")
    pool = frame.copy()
    if pool.empty:
        raise ValueError(f"{task_id} {era}: source pool is empty")
    if pool["public_sample_id"].duplicated().any():
        raise ValueError(f"{task_id} {era}: duplicate public_sample_id rows")
    if not pool["y"].isin([0, 1]).all():
        raise ValueError(f"{task_id} {era}: labels must be binary 0/1")
    if _missing_text(pool["public_patient_cluster_id"]).any():
        raise ValueError(f"{task_id} {era}: missing patient cluster IDs")
    if pool["feature_row"].isna().any() or pool["row_order"].isna().any():
        raise ValueError(f"{task_id} {era}: missing feature_row or row_order")
    pool["y"] = pool["y"].astype(np.int8)
    pool["feature_row"] = pool["feature_row"].astype(np.int64)
    pool["row_order"] = pool["row_order"].astype(np.int64)
    pool["source_era"] = era
    return pool.sort_values(
        ["row_order", "public_sample_id"], kind="stable"
    ).reset_index(drop=True)


def _frame_signature(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values("public_sample_id", kind="stable")
    payload = "\n".join(
        f"{sample}|{patient}|{int(label)}"
        for sample, patient, label in ordered[
            ["public_sample_id", "public_patient_cluster_id", "y"]
        ].itertuples(index=False, name=None)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_fixed_patient_disjoint_test(
    pre_pool: pd.DataFrame,
    current_pool: pd.DataFrame,
    test: pd.DataFrame,
    *,
    task_id: str,
) -> str:
    """Validate one fixed future test against the union of source histories."""

    missing = {
        "public_sample_id",
        "public_patient_cluster_id",
        "feature_row",
        "y",
    }.difference(test.columns)
    if missing:
        raise ValueError(f"{task_id} test: missing columns {sorted(missing)}")
    if test.empty:
        raise ValueError(f"{task_id} test: fixed future test is empty")
    if test["public_sample_id"].duplicated().any():
        raise ValueError(f"{task_id} test: duplicate public_sample_id rows")
    if not test["y"].isin([0, 1]).all() or test["y"].nunique() != 2:
        raise ValueError(f"{task_id} test: both binary classes are required")
    if _missing_text(test["public_patient_cluster_id"]).any():
        raise ValueError(f"{task_id} test: missing patient cluster IDs")

    development = pd.concat([pre_pool, current_pool], ignore_index=True)
    sample_overlap = set(development["public_sample_id"].astype(str)) & set(
        test["public_sample_id"].astype(str)
    )
    if sample_overlap:
        raise ValueError(f"{task_id} test: sample overlap with development")
    patient_overlap = set(development["public_patient_cluster_id"].astype(str)) & set(
        test["public_patient_cluster_id"].astype(str)
    )
    if patient_overlap:
        raise ValueError(f"{task_id} test: patient overlap with development")
    return _frame_signature(test)


def class_cap_table(
    pre_pool: pd.DataFrame,
    current_pool: pd.DataFrame,
    *,
    task_id: str,
) -> pd.DataFrame:
    """Return the common per-class sample cap used by all regimes."""

    rows: list[dict[str, object]] = []
    for label in (0, 1):
        pre_n = int(pre_pool["y"].eq(label).sum())
        current_n = int(current_pool["y"].eq(label).sum())
        rows.append(
            {
                "task_id": task_id,
                "class_label": label,
                "pre_marker_available_n": pre_n,
                "current_workflow_available_n": current_n,
                "common_class_cap_n": min(pre_n, current_n),
            }
        )
    return pd.DataFrame(rows)


def _permuted_pool(
    pool: pd.DataFrame,
    *,
    label: int,
    seed: int,
    task_id: str,
    repeat_index: int,
    era: str,
) -> pd.DataFrame:
    one_class = pool.loc[pool["y"].eq(label)].reset_index(drop=True)
    rng = np.random.default_rng(
        deterministic_seed(
            seed,
            ANALYSIS_ID,
            task_id,
            repeat_index,
            era,
            f"class={label}",
            "sampling_order",
        )
    )
    order = rng.permutation(len(one_class))
    return one_class.iloc[order].reset_index(drop=True)


def _pooled_era_counts(target_n: int, *, extra_to_pre: bool) -> tuple[int, int]:
    half = target_n // 2
    if target_n % 2 == 0:
        return half, half
    if extra_to_pre:
        return half + 1, half
    return half, half + 1


def build_repeated_size_matched_cells(
    pre_frame: pd.DataFrame,
    current_frame: pd.DataFrame,
    *,
    task_id: str,
    learning_fractions: Sequence[float] = REQUIRED_LEARNING_FRACTIONS,
    repeats: int = 20,
    seed: int = DEFAULT_SEED,
    minimum_train_class_n: int = 10,
    require_prespecified_grid: bool = True,
) -> tuple[list[SampledTrainingCell], pd.DataFrame, pd.DataFrame]:
    """Construct deterministic nested size-matched training cohorts.

    The common class cap is ``min(pre_n, current_n)`` separately for labels 0
    and 1. At every learning fraction, all three regimes receive exactly the
    same class counts. Within the pooled regime, each class is divided between
    eras with an imbalance of at most one row.
    """

    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if minimum_train_class_n < 1:
        raise ValueError("minimum_train_class_n must be at least 1")
    fractions = validate_learning_fractions(
        learning_fractions, require_prespecified_grid=require_prespecified_grid
    )
    pre = _canonical_pool(pre_frame, era=PRE_ERA, task_id=task_id)
    current = _canonical_pool(current_frame, era=CURRENT_ERA, task_id=task_id)
    overlap = set(pre["public_sample_id"].astype(str)) & set(
        current["public_sample_id"].astype(str)
    )
    if overlap:
        raise ValueError(f"{task_id}: source-era sample overlap")

    caps = class_cap_table(pre, current, task_id=task_id)
    cap_by_class = {
        int(row.class_label): int(row.common_class_cap_n)
        for row in caps.itertuples(index=False)
    }
    if min(cap_by_class.values()) < 1:
        raise ValueError(f"{task_id}: both source eras must contain both classes")

    cells: list[SampledTrainingCell] = []
    audit_rows: list[dict[str, object]] = []
    for repeat_index in range(repeats):
        pre_orders = {
            label: _permuted_pool(
                pre,
                label=label,
                seed=seed,
                task_id=task_id,
                repeat_index=repeat_index,
                era=PRE_ERA,
            )
            for label in (0, 1)
        }
        current_orders = {
            label: _permuted_pool(
                current,
                label=label,
                seed=seed,
                task_id=task_id,
                repeat_index=repeat_index,
                era=CURRENT_ERA,
            )
            for label in (0, 1)
        }
        for fraction in fractions:
            target = {
                label: int(np.floor(cap_by_class[label] * fraction + 1e-12))
                for label in (0, 1)
            }
            reason = ""
            status = "ok"
            if min(target.values()) < minimum_train_class_n:
                status = "insufficient"
                reason = (
                    f"fractional_min_class={min(target.values())}<"
                    f"minimum_train_class_n={minimum_train_class_n}"
                )

            regime_parts: dict[str, list[pd.DataFrame]] = {
                CURRENT_ONLY: [],
                PRE_MARKER_ONLY: [],
                POOLED_ERA_BALANCED: [],
            }
            for label in (0, 1):
                class_n = target[label]
                regime_parts[CURRENT_ONLY].append(current_orders[label].iloc[:class_n])
                regime_parts[PRE_MARKER_ONLY].append(pre_orders[label].iloc[:class_n])
                pre_n, current_n = _pooled_era_counts(
                    class_n,
                    extra_to_pre=(repeat_index + label) % 2 == 0,
                )
                regime_parts[POOLED_ERA_BALANCED].extend(
                    [
                        pre_orders[label].iloc[:pre_n],
                        current_orders[label].iloc[:current_n],
                    ]
                )

            for regime in TRAINING_REGIMES:
                selected = pd.concat(regime_parts[regime], ignore_index=True)
                selected = selected.sort_values(
                    ["row_order", "public_sample_id"], kind="stable"
                ).reset_index(drop=True)
                observed = selected["y"].value_counts().to_dict()
                if int(observed.get(0, 0)) != target[0] or int(
                    observed.get(1, 0)
                ) != target[1]:
                    raise AssertionError("Size-matched sampler produced unequal class counts")
                pre_n = int(selected["source_era"].eq(PRE_ERA).sum())
                current_n = int(selected["source_era"].eq(CURRENT_ERA).sum())
                signature = _frame_signature(selected)
                cell = SampledTrainingCell(
                    task_id=task_id,
                    repeat_index=repeat_index,
                    learning_fraction=fraction,
                    training_regime=regime,
                    frame=selected,
                    target_negative_n=target[0],
                    target_positive_n=target[1],
                    pre_era_n=pre_n,
                    current_era_n=current_n,
                    status=status,
                    insufficient_reason=reason,
                    sample_signature=signature,
                )
                cells.append(cell)
                audit_rows.append(
                    {
                        "analysis_id": ANALYSIS_ID,
                        "task_id": task_id,
                        "repeat_index": repeat_index,
                        "learning_fraction": fraction,
                        "training_regime": regime,
                        "n_development": len(selected),
                        "development_negative_n": target[0],
                        "development_positive_n": target[1],
                        "pre_era_n": pre_n,
                        "current_era_n": current_n,
                        "status": status,
                        "insufficient_reason": reason,
                        "training_sample_signature": signature,
                    }
                )
    return cells, pd.DataFrame(audit_rows), caps


def run_size_matched_task(
    inputs: TrainingHistoryInputs,
    cohorts: HistoryTaskCohorts,
    *,
    learning_fractions: Sequence[float] = REQUIRED_LEARNING_FRACTIONS,
    repeats: int = 20,
    threads: int = 4,
    seed: int = DEFAULT_SEED,
    minimum_train_class_n: int = 10,
    return_predictions: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit all size-matched cells for one task on one immutable test cohort."""

    pre = cohorts.development_by_regime["pre_marker_history_only"]
    current = cohorts.development_by_regime["current_workflow_only"]
    test = cohorts.test_patient_disjoint_common.copy()
    pre_missing_patient = _missing_text(pre["public_patient_cluster_id"])
    current_missing_patient = _missing_text(current["public_patient_cluster_id"])
    pre_missing_patient_excluded_n = int(pre_missing_patient.sum())
    current_missing_patient_excluded_n = int(current_missing_patient.sum())
    pre_validated = _canonical_pool(
        pre.loc[~pre_missing_patient].copy(),
        era=PRE_ERA,
        task_id=cohorts.task_id,
    )
    current_validated = _canonical_pool(
        current.loc[~current_missing_patient].copy(),
        era=CURRENT_ERA,
        task_id=cohorts.task_id,
    )
    fixed_test_signature = validate_fixed_patient_disjoint_test(
        pre_validated,
        current_validated,
        test,
        task_id=cohorts.task_id,
    )
    cells, sampling_audit, caps = build_repeated_size_matched_cells(
        pre_validated,
        current_validated,
        task_id=cohorts.task_id,
        learning_fractions=learning_fractions,
        repeats=repeats,
        seed=seed,
        minimum_train_class_n=minimum_train_class_n,
    )
    sampling_audit["pre_missing_patient_excluded_n"] = (
        pre_missing_patient_excluded_n
    )
    sampling_audit["current_missing_patient_excluded_n"] = (
        current_missing_patient_excluded_n
    )
    caps["pre_missing_patient_excluded_n"] = pre_missing_patient_excluded_n
    caps["current_missing_patient_excluded_n"] = (
        current_missing_patient_excluded_n
    )

    params = dict(inputs.fixed_parameters[cohorts.task_id])
    x_test, y_test = matrix_rows(inputs.matrix, test)
    test_positive_n = int(y_test.sum())
    test_negative_n = int(len(y_test) - test_positive_n)
    test_baseline = float(np.mean(y_test))
    test_patient_n = int(test["public_patient_cluster_id"].nunique())
    metric_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []

    for cell in cells:
        model_seed = deterministic_seed(
            seed,
            ANALYSIS_ID,
            cohorts.task_id,
            cell.repeat_index,
            f"fraction={cell.learning_fraction:.8f}",
            "shared_across_regimes",
        )
        row: dict[str, object] = {
            "analysis_id": ANALYSIS_ID,
            "task_id": cohorts.task_id,
            "training_regime": cell.training_regime,
            "repeat_index": cell.repeat_index,
            "learning_fraction": cell.learning_fraction,
            "model": "lightgbm",
            "endpoint": "historical_S_vs_IR",
            "feature_representation": "intensity6000",
            "status": cell.status,
            "insufficient_reason": cell.insufficient_reason,
            "n_development": len(cell.frame),
            "development_positive_n": cell.target_positive_n,
            "development_negative_n": cell.target_negative_n,
            "pre_era_n": cell.pre_era_n,
            "current_era_n": cell.current_era_n,
            "pre_missing_patient_excluded_n": pre_missing_patient_excluded_n,
            "current_missing_patient_excluded_n": (
                current_missing_patient_excluded_n
            ),
            "training_sample_signature": cell.sample_signature,
            "n_test": len(test),
            "test_positive_n": test_positive_n,
            "test_negative_n": test_negative_n,
            "test_patient_cluster_n": test_patient_n,
            "fixed_test_signature": fixed_test_signature,
            "raw_auroc": np.nan,
            "raw_auprc": np.nan,
            "raw_auprc_baseline": test_baseline,
            "raw_auprc_lift": np.nan,
            "model_seed": model_seed,
            "hyperparameter_source": "frozen_primary_metrics_same_task",
            "best_hyperparameters": json.dumps(params, sort_keys=True),
            "test_labels_used_for_sampling_or_tuning": False,
            "calibration_applied": False,
            "threshold_selection_applied": False,
        }
        if cell.status == "ok":
            x_development, y_development = matrix_rows(inputs.matrix, cell.frame)
            model = fit_lightgbm(
                params,
                x_development,
                y_development,
                model_seed,
                threads,
            )
            probability = predict(model, x_test)
            raw_auroc = float(roc_auc_score(y_test, probability))
            raw_auprc = float(average_precision_score(y_test, probability))
            row.update(
                {
                    "raw_auroc": raw_auroc,
                    "raw_auprc": raw_auprc,
                    "raw_auprc_lift": (
                        raw_auprc / test_baseline if test_baseline > 0 else np.nan
                    ),
                }
            )
            if return_predictions:
                prediction = test[
                    ["public_sample_id", "public_patient_cluster_id", "y"]
                ].copy()
                prediction.insert(0, "learning_fraction", cell.learning_fraction)
                prediction.insert(0, "repeat_index", cell.repeat_index)
                prediction.insert(0, "training_regime", cell.training_regime)
                prediction.insert(0, "task_id", cohorts.task_id)
                prediction["raw_probability"] = probability
                prediction["fixed_test_signature"] = fixed_test_signature
                prediction_parts.append(prediction)
        metric_rows.append(row)

    predictions = (
        pd.concat(prediction_parts, ignore_index=True)
        if prediction_parts
        else pd.DataFrame()
    )
    return pd.DataFrame(metric_rows), sampling_audit, caps, predictions


def paired_size_matched_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    """Calculate paired raw-metric deltas for each repeat and fraction."""

    required = {
        "task_id",
        "training_regime",
        "repeat_index",
        "learning_fraction",
        "status",
        "n_development",
        "development_positive_n",
        "development_negative_n",
        "n_test",
        "fixed_test_signature",
        "raw_auroc",
        "raw_auprc",
    }
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Metrics missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    keys = ["task_id", "repeat_index", "learning_fraction"]
    for key, group in metrics.groupby(keys, sort=True, dropna=False):
        task_id, repeat_index, fraction = key
        successful = group.loc[group["status"].eq("ok")].copy()
        if set(successful["training_regime"]) != set(TRAINING_REGIMES):
            continue
        if successful["training_regime"].duplicated().any():
            raise ValueError(f"{key}: duplicate training-regime metric rows")
        if successful["fixed_test_signature"].nunique() != 1 or successful[
            "n_test"
        ].nunique() != 1:
            raise ValueError(f"{key}: training regimes do not share one fixed test")
        for column in (
            "n_development",
            "development_positive_n",
            "development_negative_n",
        ):
            if successful[column].nunique() != 1:
                raise ValueError(f"{key}: regimes are not size/class matched ({column})")
        indexed = successful.set_index("training_regime")
        for comparator, reference in PAIRED_COMPARISONS:
            for metric_name in ("raw_auroc", "raw_auprc"):
                comparator_value = float(indexed.at[comparator, metric_name])
                reference_value = float(indexed.at[reference, metric_name])
                rows.append(
                    {
                        "analysis_id": ANALYSIS_ID,
                        "task_id": task_id,
                        "repeat_index": int(repeat_index),
                        "learning_fraction": float(fraction),
                        "comparison_id": f"{comparator}_minus_{reference}",
                        "comparator_regime": comparator,
                        "reference_regime": reference,
                        "metric": metric_name,
                        "comparator_value": comparator_value,
                        "reference_value": reference_value,
                        "delta_comparator_minus_reference": (
                            comparator_value - reference_value
                        ),
                        "n_development_per_regime": int(
                            successful["n_development"].iloc[0]
                        ),
                        "development_positive_n_per_regime": int(
                            successful["development_positive_n"].iloc[0]
                        ),
                        "development_negative_n_per_regime": int(
                            successful["development_negative_n"].iloc[0]
                        ),
                        "n_test": int(successful["n_test"].iloc[0]),
                        "fixed_test_signature": successful[
                            "fixed_test_signature"
                        ].iloc[0],
                    }
                )
    return pd.DataFrame(rows, columns=PAIRED_DELTA_COLUMNS)


def summarize_paired_deltas(deltas: pd.DataFrame) -> pd.DataFrame:
    """Summarize the repeated-sampling delta distribution without calling it a CI."""

    if deltas.empty:
        return pd.DataFrame(columns=DELTA_SUMMARY_COLUMNS)
    required = {
        "task_id",
        "learning_fraction",
        "comparison_id",
        "comparator_regime",
        "reference_regime",
        "metric",
        "delta_comparator_minus_reference",
    }
    missing = required.difference(deltas.columns)
    if missing:
        raise ValueError(f"Deltas missing columns: {sorted(missing)}")
    group_columns = [
        "task_id",
        "learning_fraction",
        "comparison_id",
        "comparator_regime",
        "reference_regime",
        "metric",
    ]
    rows: list[dict[str, object]] = []
    for key, group in deltas.groupby(group_columns, sort=True, dropna=False):
        values = group["delta_comparator_minus_reference"].to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(group_columns, key, strict=True)),
                "repeat_n": len(values),
                "delta_mean": float(np.mean(values)),
                "delta_median": float(np.median(values)),
                "delta_sd": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "repeat_distribution_q025": float(np.quantile(values, 0.025)),
                "repeat_distribution_q975": float(np.quantile(values, 0.975)),
                "fraction_delta_gt_zero": float(np.mean(values > 0)),
            }
        )
    return pd.DataFrame(rows, columns=DELTA_SUMMARY_COLUMNS)
