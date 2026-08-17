"""Same-test training-history comparison for the ZD-MAST major revision."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .cross_platform import (
    TASK_IDS,
    _bootstrap_groups,
    _cluster_bootstrap_intervals,
    _resample_cluster_indices,
    deterministic_seed,
    fit_guarded_platt,
    probability_metrics,
    select_fixed_thresholds,
)
from .modeling import (
    PublicProtocolData,
    fit_lightgbm,
    matrix_rows,
    out_of_fold_predictions,
    predict,
)
from .temporal_revision import build_year_folds, purge_by_training_patients, resolve_feature_root


ANALYSIS_ID = "same_test_training_history"
DEFAULT_SEED = 20260815
PRE_MARKER_END = pd.Timestamp("2025-06-22")
CURRENT_START = pd.Timestamp("2025-07-01")
CURRENT_END = pd.Timestamp("2026-02-28")
TEST_START = pd.Timestamp("2026-03-01")
TEST_END = pd.Timestamp("2026-06-09")


@dataclass(frozen=True)
class TrainingHistoryInputs:
    feature_root: Path
    matrix: np.ndarray
    bridge: pd.DataFrame
    current_folds: pd.DataFrame
    fixed_parameters: Mapping[str, dict[str, object]]


@dataclass(frozen=True)
class HistoryTaskCohorts:
    task_id: str
    development_by_regime: Mapping[str, pd.DataFrame]
    folds_by_regime: Mapping[str, list[tuple[pd.DataFrame, pd.DataFrame, str]]]
    fold_audit: pd.DataFrame
    test_all_samples: pd.DataFrame
    test_patient_disjoint_common: pd.DataFrame
    test_purge_audit: dict[str, object]


def load_training_history_inputs(
    release_root: Path,
    temporal_bridge_path: Path,
    primary_metrics_path: Path,
) -> TrainingHistoryInputs:
    feature = resolve_feature_root(release_root)
    matrix = np.load(feature / "zd_mast_a_sample_level_intensity6000_v1.0.0.npy", mmap_mode="r")
    if matrix.shape[1] != 6000:
        raise ValueError(f"Expected intensity6000, found {matrix.shape}")
    bridge = pd.read_parquet(temporal_bridge_path).copy()
    required = {
        "public_sample_id",
        "task_id",
        "collection_date",
        "historical_binary_S_vs_IR",
        "public_patient_cluster_id",
        "release_feature_row",
    }
    missing = required.difference(bridge.columns)
    if missing:
        raise ValueError(f"Temporal bridge missing columns: {sorted(missing)}")
    bridge["collection_date"] = pd.to_datetime(bridge["collection_date"], errors="raise")
    bridge["year"] = bridge["collection_date"].dt.year.astype(int)
    bridge["y"] = bridge["historical_binary_S_vs_IR"].astype(np.int8)
    bridge["feature_row"] = bridge["release_feature_row"].astype(int)
    bridge = bridge.sort_values(
        ["collection_date", "public_sample_id", "task_id"], kind="stable"
    ).reset_index(drop=True)
    bridge["row_order"] = np.arange(len(bridge), dtype=np.int64)
    if bridge.duplicated(["public_sample_id", "task_id"]).any():
        raise ValueError("Temporal bridge contains duplicate sample-task rows")
    if bridge["feature_row"].min() < 0 or bridge["feature_row"].max() >= matrix.shape[0]:
        raise ValueError("Temporal bridge feature rows are out of bounds")

    current_folds = pd.read_csv(
        feature / "zd_mast_protocol_b_rolling_origin_folds_public_v1.0.0.csv"
    )
    current_folds = current_folds.loc[
        current_folds["analysis_id"].eq("local_temporal")
        & current_folds["protocol"].eq("current_workflow_protocol_b")
        & current_folds["task_id"].isin(TASK_IDS)
    ].copy()
    metrics = pd.read_csv(primary_metrics_path)
    if "task_id" not in metrics or "best_hyperparameters" not in metrics:
        raise ValueError("Primary metrics lack task_id/best_hyperparameters")
    if metrics["task_id"].duplicated().any():
        raise ValueError("Primary metrics contain duplicate task rows")
    params = {
        str(row.task_id): json.loads(row.best_hyperparameters)
        for row in metrics.itertuples(index=False)
    }
    if set(TASK_IDS) - set(params):
        raise ValueError(f"Missing fixed parameters for {sorted(set(TASK_IDS) - set(params))}")
    return TrainingHistoryInputs(feature, matrix, bridge, current_folds, params)


def _current_patient_purged_folds(
    task: pd.DataFrame,
    fold_table: pd.DataFrame,
    pre_history: pd.DataFrame | None = None,
) -> tuple[list[tuple[pd.DataFrame, pd.DataFrame, str]], pd.DataFrame]:
    folds: list[tuple[pd.DataFrame, pd.DataFrame, str]] = []
    audits: list[dict[str, object]] = []
    task_for_merge = task.rename(columns={"row_order": "temporal_row_order"})
    for fold_index in sorted(fold_table["fold_index"].unique()):
        one = fold_table.loc[fold_table["fold_index"].eq(fold_index)].copy()
        one = one.merge(
            task_for_merge,
            on=["public_sample_id", "task_id"],
            how="left",
            validate="many_to_one",
        )
        if one["y"].isna().any():
            raise ValueError(f"Current fold {fold_index} references an unmapped temporal row")
        if not one["collection_date"].between(
            CURRENT_START, CURRENT_END, inclusive="both"
        ).all():
            raise ValueError(f"Current fold {fold_index} contains rows outside the frozen window")
        train = one.loc[one["split"].eq("train")].copy()
        if pre_history is not None:
            earlier = pre_history.rename(columns={"row_order": "temporal_row_order"})
            train = pd.concat([earlier, train], ignore_index=True)
            train = train.drop_duplicates(["public_sample_id", "task_id"], keep="last")
        validation_before = one.loc[one["split"].eq("validation")].copy()
        validation, purge = purge_by_training_patients(train, validation_before)
        train_counts = train["y"].value_counts()
        validation_counts = validation["y"].value_counts()
        adequate = (
            len(train) >= 20
            and train_counts.size == 2
            and int(train_counts.min()) >= 10
            and len(validation) >= 10
            and validation_counts.size == 2
            and int(validation_counts.min()) >= 5
        )
        audits.append(
            {
                "fold_index": int(fold_index),
                "fold_type": "current_workflow_rolling_origin",
                "n_train": len(train),
                "train_positive_n": int(train["y"].sum()),
                "train_negative_n": int(train["y"].eq(0).sum()),
                "n_validation_before_purge": len(validation_before),
                "removed_patient_overlap_n": purge["removed_patient_overlap_n"],
                "removed_missing_patient_cluster_n": purge[
                    "removed_missing_patient_cluster_n"
                ],
                "n_validation": len(validation),
                "validation_positive_n": int(validation["y"].sum()),
                "validation_negative_n": int(validation["y"].eq(0).sum()),
                "status": "adequate" if adequate else "insufficient",
            }
        )
        if adequate:
            train["row_order"] = train["temporal_row_order"].astype(int)
            validation["row_order"] = validation["temporal_row_order"].astype(int)
            folds.append(
                (
                    train.sort_values("row_order", kind="stable").reset_index(drop=True),
                    validation.sort_values("row_order", kind="stable").reset_index(drop=True),
                    f"current_fold={int(fold_index)}",
                )
            )
    return folds, pd.DataFrame(audits)


def build_history_cohorts(inputs: TrainingHistoryInputs, task_id: str) -> HistoryTaskCohorts:
    task = inputs.bridge.loc[inputs.bridge["task_id"].eq(task_id)].copy()
    pre = task.loc[task["collection_date"].le(PRE_MARKER_END)].copy()
    current = task.loc[
        task["collection_date"].between(CURRENT_START, CURRENT_END, inclusive="both")
    ].copy()
    pooled = pd.concat([pre, current], ignore_index=True).drop_duplicates(
        ["public_sample_id", "task_id"], keep="last"
    )
    test = task.loc[
        task["collection_date"].between(TEST_START, TEST_END, inclusive="both")
    ].copy()
    development_by_regime = {
        "pre_marker_history_only": pre.sort_values("row_order", kind="stable").reset_index(drop=True),
        "current_workflow_only": current.sort_values("row_order", kind="stable").reset_index(drop=True),
        "pooled_pre_and_current": pooled.sort_values("row_order", kind="stable").reset_index(drop=True),
    }
    common_test, purge = purge_by_training_patients(pooled, test)

    pre_folds, pre_audit = build_year_folds(pre)
    pre_audit.insert(0, "regime", "pre_marker_history_only")
    fold_table = inputs.current_folds.loc[inputs.current_folds["task_id"].eq(task_id)]
    current_folds, current_audit = _current_patient_purged_folds(task, fold_table)
    current_audit.insert(0, "regime", "current_workflow_only")
    pooled_folds, pooled_audit = _current_patient_purged_folds(task, fold_table, pre)
    pooled_audit.insert(0, "regime", "pooled_pre_and_current")
    fold_audit = pd.concat([pre_audit, current_audit, pooled_audit], ignore_index=True, sort=False)
    fold_audit.insert(0, "task_id", task_id)
    return HistoryTaskCohorts(
        task_id=task_id,
        development_by_regime=development_by_regime,
        folds_by_regime={
            "pre_marker_history_only": pre_folds,
            "current_workflow_only": current_folds,
            "pooled_pre_and_current": pooled_folds,
        },
        fold_audit=fold_audit,
        test_all_samples=test.sort_values("row_order", kind="stable").reset_index(drop=True),
        test_patient_disjoint_common=common_test.sort_values("row_order", kind="stable").reset_index(drop=True),
        test_purge_audit=purge,
    )


def run_history_task(
    inputs: TrainingHistoryInputs,
    cohorts: HistoryTaskCohorts,
    *,
    threads: int,
    bootstrap_count: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    bootstraps: list[pd.DataFrame] = []
    predictions: list[pd.DataFrame] = []
    params = inputs.fixed_parameters[cohorts.task_id]
    common_task_seed = deterministic_seed(seed, ANALYSIS_ID, cohorts.task_id, "shared_model_seed")
    for regime, development in cohorts.development_by_regime.items():
        folds = cohorts.folds_by_regime[regime]
        if len(folds) < 2:
            raise ValueError(f"{cohorts.task_id} {regime}: fewer than two valid folds")
        data = PublicProtocolData(
            task_id=cohorts.task_id,
            feature_matrix=inputs.matrix,
            development=development,
            test=cohorts.test_patient_disjoint_common,
            folds=folds,
        )
        regime_seed = deterministic_seed(seed, ANALYSIS_ID, cohorts.task_id, regime)
        oof = out_of_fold_predictions(data, params, common_task_seed + 10_000, threads)
        oof_y = oof["y"].to_numpy(dtype=np.int8)
        calibrator = fit_guarded_platt(oof_y, oof["raw_probability"].to_numpy(dtype=float))
        calibrated_oof = calibrator.apply(oof["raw_probability"].to_numpy(dtype=float))
        thresholds = (
            select_fixed_thresholds(oof_y, calibrated_oof)
            if calibrated_oof is not None
            else {}
        )
        x_dev, y_dev = matrix_rows(inputs.matrix, development)
        model = fit_lightgbm(params, x_dev, y_dev, common_task_seed + 20_000, threads)
        for variant, test in (
            ("patient_disjoint_common_test_primary", cohorts.test_patient_disjoint_common),
            ("all_sample_common_test_sensitivity", cohorts.test_all_samples),
        ):
            x_test, y_test = matrix_rows(inputs.matrix, test)
            raw = predict(model, x_test)
            calibrated = calibrator.apply(raw)
            metrics = probability_metrics(y_test, raw, calibrated, thresholds)
            groups, group_source = _bootstrap_groups(test)
            row: dict[str, object] = {
                "analysis_id": ANALYSIS_ID,
                "task_id": cohorts.task_id,
                "training_regime": regime,
                "analysis_variant": variant,
                "model": "lightgbm",
                "endpoint": "historical_S_vs_IR",
                "feature_representation": "intensity6000",
                "status": "ok",
                "n_development": len(development),
                "development_positive_n": int(y_dev.sum()),
                "development_negative_n": int(y_dev.size - y_dev.sum()),
                "valid_fold_n": len(folds),
                "oof_n": len(oof),
                "n_test": len(test),
                "test_positive_n": int(y_test.sum()),
                "test_negative_n": int(y_test.size - y_test.sum()),
                "patient_cluster_n": int(pd.Series(groups).nunique()),
                "bootstrap_group_source": group_source,
                "calibration_status": calibrator.status,
                "calibration_failure_reason": calibrator.reason,
                "oof_platt_slope": calibrator.slope,
                "oof_platt_intercept": calibrator.intercept,
                "hyperparameter_source": "frozen_current_workflow_protocol_b_same_task",
                "shared_model_seed_across_training_regimes": common_task_seed + 20_000,
                "best_hyperparameters": json.dumps(params, sort_keys=True),
                "test_labels_used_for_tuning": False,
                **metrics,
            }
            if bootstrap_count > 0:
                boot = _cluster_bootstrap_intervals(
                    y_test,
                    raw,
                    calibrated,
                    groups,
                    thresholds,
                    n_boot=bootstrap_count,
                    seed=deterministic_seed(regime_seed, variant, "bootstrap"),
                )
                if not boot.empty:
                    boot.insert(0, "analysis_variant", variant)
                    boot.insert(0, "training_regime", regime)
                    boot.insert(0, "task_id", cohorts.task_id)
                    bootstraps.append(boot)
                    for interval in boot.itertuples(index=False):
                        row[f"{interval.metric}_ci_low"] = interval.ci_low
                        row[f"{interval.metric}_ci_high"] = interval.ci_high
            prediction = test[
                ["public_sample_id", "public_patient_cluster_id", "y"]
            ].copy()
            prediction.insert(0, "analysis_variant", variant)
            prediction.insert(0, "training_regime", regime)
            prediction.insert(0, "task_id", cohorts.task_id)
            prediction["raw_probability"] = raw
            prediction["calibrated_probability"] = calibrated if calibrated is not None else np.nan
            predictions.append(prediction)
            rows.append(row)
    return (
        pd.DataFrame(rows),
        pd.concat(bootstraps, ignore_index=True) if bootstraps else pd.DataFrame(),
        pd.concat(predictions, ignore_index=True),
    )


def paired_history_deltas(
    predictions: pd.DataFrame,
    *,
    bootstrap_count: int,
    seed: int,
) -> pd.DataFrame:
    primary = predictions.loc[
        predictions["analysis_variant"].eq("patient_disjoint_common_test_primary")
    ].copy()
    rows: list[dict[str, object]] = []
    for task_id, task in primary.groupby("task_id"):
        pivot = task.pivot(
            index=["public_sample_id", "public_patient_cluster_id", "y"],
            columns="training_regime",
            values="raw_probability",
        ).reset_index()
        required = {"pre_marker_history_only", "current_workflow_only", "pooled_pre_and_current"}
        if not required.issubset(pivot.columns) or pivot[list(required)].isna().any().any():
            raise ValueError(f"{task_id}: prediction rows are not aligned across training regimes")
        y = pivot["y"].to_numpy(dtype=np.int8)
        groups = pivot["public_patient_cluster_id"].astype(str).to_numpy()
        for comparator in ("pre_marker_history_only", "pooled_pre_and_current"):
            for metric_name, metric in (
                ("raw_auroc", roc_auc_score),
                ("raw_auprc", average_precision_score),
            ):
                current = pivot["current_workflow_only"].to_numpy(dtype=float)
                comparison = pivot[comparator].to_numpy(dtype=float)
                point = float(metric(y, comparison) - metric(y, current))
                rng = np.random.default_rng(
                    deterministic_seed(seed, task_id, comparator, metric_name)
                )
                draws = []
                for _ in range(bootstrap_count):
                    index = _resample_cluster_indices(groups, rng)
                    if np.unique(y[index]).size < 2:
                        continue
                    draws.append(float(metric(y[index], comparison[index]) - metric(y[index], current[index])))
                values = np.asarray(draws, dtype=float)
                rows.append(
                    {
                        "task_id": task_id,
                        "comparator_regime": comparator,
                        "reference_regime": "current_workflow_only",
                        "metric": metric_name,
                        "delta_comparator_minus_current": point,
                        "bootstrap_requested_n": bootstrap_count,
                        "bootstrap_valid_n": len(values),
                        "ci_low": float(np.quantile(values, 0.025)),
                        "ci_high": float(np.quantile(values, 0.975)),
                    }
                )
    return pd.DataFrame(rows)
