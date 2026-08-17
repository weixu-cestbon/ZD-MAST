"""Reviewer-grade longitudinal temporal analyses for ZD-MAST.

The module retains the frozen historical ten-task panel and historical S versus
I/R endpoint. Calendar-year tests use only earlier years for development,
patient-purged rolling-origin validation, and a patient-disjoint primary test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .cross_platform import (
    TASK_IDS,
    _bootstrap_groups,
    _cluster_bootstrap_intervals,
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
    tune_with_frozen_folds,
)


TEST_YEARS: tuple[int, ...] = (2022, 2023, 2024, 2025)
ANALYSIS_ID = "calendar_year_rolling_patient_purged"
DEFAULT_SEED = 20260815


@dataclass(frozen=True)
class AnnualInputs:
    feature_root: Path
    matrix: np.ndarray
    base: pd.DataFrame
    splits: pd.DataFrame


@dataclass(frozen=True)
class AnnualCohorts:
    task_id: str
    test_year: int
    development: pd.DataFrame
    test_all_samples: pd.DataFrame
    test_patient_disjoint: pd.DataFrame
    folds: list[tuple[pd.DataFrame, pd.DataFrame, str]]
    fold_audit: pd.DataFrame
    test_purge_audit: dict[str, object]


def resolve_feature_root(release_root: Path) -> Path:
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


def load_annual_inputs(release_root: Path) -> AnnualInputs:
    feature = resolve_feature_root(release_root)
    matrix = np.load(feature / "zd_mast_a_sample_level_intensity6000_v1.0.0.npy", mmap_mode="r")
    if matrix.ndim != 2 or matrix.shape[1] != 6000:
        raise ValueError(f"Expected Site A intensity6000 matrix, found shape {matrix.shape}")
    sample = pd.read_csv(feature / "zd_mast_sample_metadata_public_v1.0.0.csv")
    sample = sample.loc[sample["site_id"].eq("ZD-MAST-A"), ["public_sample_id", "feature_row"]]
    labels = pd.read_parquet(feature / "zd_mast_ast_labels_historical_v1.0.0.parquet")
    labels = labels.loc[
        labels["site_id"].eq("ZD-MAST-A")
        & labels["task_id"].isin(TASK_IDS)
        & labels["binary_s_vs_ir"].isin([0, 1]),
        ["public_sample_id", "task_id", "binary_s_vs_ir", "year"],
    ].copy()
    labels["y"] = labels.pop("binary_s_vs_ir").astype(np.int8)
    groups = pd.read_parquet(feature / "zd_mast_patient_episode_groups_public_v1.0.0.parquet")
    required_groups = [
        "public_sample_id",
        "task_id",
        "public_patient_cluster_id",
        "public_episode_id",
        "episode_first_sample_flag",
    ]
    if groups.duplicated(["public_sample_id", "task_id"]).any():
        raise ValueError("Patient grouping table has duplicate sample-task rows")
    base = labels.merge(sample, on="public_sample_id", how="inner", validate="many_to_one")
    base = base.merge(
        groups[required_groups],
        on=["public_sample_id", "task_id"],
        how="left",
        validate="one_to_one",
    )
    if base.duplicated(["public_sample_id", "task_id"]).any():
        raise ValueError("Site A annual base contains duplicate sample-task rows")
    if base["feature_row"].min() < 0 or base["feature_row"].max() >= matrix.shape[0]:
        raise ValueError("Site A annual feature rows are out of bounds")
    splits = pd.read_csv(feature / "zd_mast_split_assignments_public_v1.0.0.csv")
    splits = splits.loc[
        splits["analysis_id"].eq("local_temporal")
        & splits["site_id"].eq("ZD-MAST-A")
        & splits["protocol"].isin([f"rolling_train_past_test_{year}" for year in TEST_YEARS])
        & splits["task_id"].isin(TASK_IDS)
        & splits["split"].isin(["train", "test"])
    ].copy()
    if splits.duplicated(["protocol", "task_id", "public_sample_id"]).any():
        raise ValueError("Annual split table has duplicate task-protocol assignments")
    return AnnualInputs(feature, matrix, base, splits)


def support_status(frame: pd.DataFrame, minimum_total: int, minimum_class: int) -> tuple[str, str]:
    if frame.empty:
        return "absent", "no_split_rows"
    counts = frame["y"].value_counts()
    if len(frame) < minimum_total:
        return "insufficient", f"n<{minimum_total}"
    if counts.size < 2:
        return "insufficient", "single_class"
    if int(counts.min()) < minimum_class:
        return "insufficient", f"min_class<{minimum_class}"
    return "adequate", ""


def purge_by_training_patients(
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    train_patients = set(training["public_patient_cluster_id"].dropna().astype(str))
    patient = evaluation["public_patient_cluster_id"].astype("string")
    missing = patient.isna() | patient.str.strip().eq("")
    overlap = patient.fillna("").astype(str).isin(train_patients)
    keep = ~missing & ~overlap
    purged = evaluation.loc[keep].copy().reset_index(drop=True)
    return purged, {
        "n_before": int(len(evaluation)),
        "removed_patient_overlap_n": int(overlap.sum()),
        "removed_missing_patient_cluster_n": int(missing.sum()),
        "n_after": int(len(purged)),
    }


def build_year_folds(
    development: pd.DataFrame,
    *,
    minimum_train_total: int = 20,
    minimum_train_class: int = 10,
    minimum_validation_total: int = 10,
    minimum_validation_class: int = 5,
) -> tuple[list[tuple[pd.DataFrame, pd.DataFrame, str]], pd.DataFrame]:
    folds: list[tuple[pd.DataFrame, pd.DataFrame, str]] = []
    audit_rows: list[dict[str, object]] = []
    years = sorted(int(year) for year in development["year"].dropna().unique())
    for validation_year in years[1:]:
        train = development.loc[development["year"].lt(validation_year)].copy()
        validation_before = development.loc[development["year"].eq(validation_year)].copy()
        validation, purge = purge_by_training_patients(train, validation_before)
        train_status, train_reason = support_status(
            train, minimum_train_total, minimum_train_class
        )
        validation_status, validation_reason = support_status(
            validation, minimum_validation_total, minimum_validation_class
        )
        status = "adequate" if train_status == validation_status == "adequate" else "insufficient"
        reason = ";".join(value for value in (train_reason, validation_reason) if value)
        audit_rows.append(
            {
                "validation_year": validation_year,
                "train_year_min": int(train["year"].min()) if len(train) else np.nan,
                "train_year_max": int(train["year"].max()) if len(train) else np.nan,
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
                "status": status,
                "insufficient_reason": reason,
            }
        )
        if status == "adequate":
            train = train.sort_values(["year", "row_order"], kind="stable").reset_index(drop=True)
            validation = validation.sort_values(["year", "row_order"], kind="stable").reset_index(drop=True)
            folds.append((train, validation, f"validation_year={validation_year}"))
    return folds, pd.DataFrame(audit_rows)


def build_annual_cohort(inputs: AnnualInputs, task_id: str, test_year: int) -> AnnualCohorts:
    protocol = f"rolling_train_past_test_{test_year}"
    assigned = inputs.splits.loc[
        inputs.splits["protocol"].eq(protocol) & inputs.splits["task_id"].eq(task_id)
    ].copy()
    if assigned.empty:
        empty = inputs.base.iloc[0:0].assign(row_order=pd.Series(dtype=int))
        return AnnualCohorts(
            task_id,
            test_year,
            empty.copy(),
            empty.copy(),
            empty.copy(),
            [],
            pd.DataFrame(),
            {
                "n_before": 0,
                "removed_patient_overlap_n": 0,
                "removed_missing_patient_cluster_n": 0,
                "n_after": 0,
            },
        )
    assigned = assigned.merge(
        inputs.base.loc[inputs.base["task_id"].eq(task_id)],
        on=["public_sample_id", "task_id"],
        how="left",
        validate="one_to_one",
    )
    if assigned["y"].isna().any():
        raise ValueError(f"{task_id} {test_year}: split rows missing label/feature data")
    if assigned.loc[assigned["split"].eq("train"), "year"].ge(test_year).any():
        raise ValueError(f"{task_id} {test_year}: future year entered development")
    if assigned.loc[assigned["split"].eq("test"), "year"].ne(test_year).any():
        raise ValueError(f"{task_id} {test_year}: test rows are not from the named year")
    development = assigned.loc[assigned["split"].eq("train")].copy()
    test_all = assigned.loc[assigned["split"].eq("test")].copy()
    development = development.sort_values(["year", "row_order"], kind="stable").reset_index(drop=True)
    test_all = test_all.sort_values("row_order", kind="stable").reset_index(drop=True)
    test_disjoint, purge = purge_by_training_patients(development, test_all)
    folds, fold_audit = build_year_folds(development)
    return AnnualCohorts(
        task_id,
        test_year,
        development,
        test_all,
        test_disjoint,
        folds,
        fold_audit,
        purge,
    )


def _empty_result(
    cohorts: AnnualCohorts,
    variant: str,
    status: str,
    reason: str,
) -> dict[str, object]:
    test = (
        cohorts.test_patient_disjoint
        if variant == "patient_disjoint_primary"
        else cohorts.test_all_samples
    )
    return {
        "analysis_id": ANALYSIS_ID,
        "task_id": cohorts.task_id,
        "test_year": cohorts.test_year,
        "analysis_variant": variant,
        "model": "lightgbm",
        "endpoint": "historical_S_vs_IR",
        "feature_representation": "intensity6000",
        "status": status,
        "insufficient_reason": reason,
        "n_development": len(cohorts.development),
        "development_positive_n": int(cohorts.development["y"].sum()),
        "development_negative_n": int(cohorts.development["y"].eq(0).sum()),
        "n_test": len(test),
        "test_positive_n": int(test["y"].sum()),
        "test_negative_n": int(test["y"].eq(0).sum()),
        "valid_fold_n": len(cohorts.folds),
        "test_labels_used_for_tuning": False,
    }


def run_annual_cell(
    inputs: AnnualInputs,
    cohorts: AnnualCohorts,
    *,
    threads: int,
    bootstrap_count: int,
    seed: int,
) -> tuple[list[dict[str, object]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_status, all_reason = support_status(cohorts.test_all_samples, 100, 20)
    primary_status, primary_reason = support_status(cohorts.test_patient_disjoint, 100, 20)
    development_status, development_reason = support_status(cohorts.development, 100, 20)
    if all_status != "adequate" or development_status != "adequate" or len(cohorts.folds) < 2:
        status = "absent" if all_status == "absent" else "insufficient"
        reason = ";".join(
            value
            for value in (
                all_reason,
                development_reason,
                "valid_folds<2" if len(cohorts.folds) < 2 else "",
            )
            if value
        )
        return (
            [
                _empty_result(cohorts, "patient_disjoint_primary", status, reason),
                _empty_result(cohorts, "all_sample_sensitivity", status, reason),
            ],
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
    if primary_status != "adequate":
        raise ValueError(
            f"{cohorts.task_id} {cohorts.test_year}: all-sample adequate but patient-disjoint "
            f"primary is not ({primary_reason})"
        )

    data = PublicProtocolData(
        task_id=cohorts.task_id,
        feature_matrix=inputs.matrix,
        development=cohorts.development,
        test=cohorts.test_patient_disjoint,
        folds=cohorts.folds,
    )
    cell_seed = deterministic_seed(seed, ANALYSIS_ID, cohorts.task_id, cohorts.test_year)
    params, tuning = tune_with_frozen_folds(data, cell_seed, threads)
    oof = out_of_fold_predictions(data, params, cell_seed + 10_000, threads)
    oof_y = oof["y"].to_numpy(dtype=np.int8)
    calibrator = fit_guarded_platt(oof_y, oof["raw_probability"].to_numpy(dtype=float))
    calibrated_oof = calibrator.apply(oof["raw_probability"].to_numpy(dtype=float))
    thresholds = (
        select_fixed_thresholds(oof_y, calibrated_oof)
        if calibrated_oof is not None
        else {}
    )
    x_development, y_development = matrix_rows(inputs.matrix, cohorts.development)
    model = fit_lightgbm(params, x_development, y_development, cell_seed + 20_000, threads)

    rows: list[dict[str, object]] = []
    bootstraps: list[pd.DataFrame] = []
    predictions: list[pd.DataFrame] = []
    for variant, test in (
        ("patient_disjoint_primary", cohorts.test_patient_disjoint),
        ("all_sample_sensitivity", cohorts.test_all_samples),
    ):
        x_test, y_test = matrix_rows(inputs.matrix, test)
        raw = predict(model, x_test)
        calibrated = calibrator.apply(raw)
        metrics = probability_metrics(y_test, raw, calibrated, thresholds)
        row = {
            **_empty_result(cohorts, variant, "ok", ""),
            "oof_n": len(oof),
            "oof_positive_n": int(oof_y.sum()),
            "oof_negative_n": int(oof_y.size - oof_y.sum()),
            "calibration_status": calibrator.status,
            "calibration_failure_reason": calibrator.reason,
            "oof_platt_slope": calibrator.slope,
            "oof_platt_intercept": calibrator.intercept,
            "best_hyperparameters": json.dumps(params, sort_keys=True),
            **metrics,
        }
        groups, group_source = _bootstrap_groups(test)
        row["bootstrap_group_source"] = group_source
        row["patient_cluster_n"] = int(pd.Series(groups).nunique())
        if bootstrap_count > 0:
            boot = _cluster_bootstrap_intervals(
                y_test,
                raw,
                calibrated,
                groups,
                thresholds,
                n_boot=bootstrap_count,
                seed=deterministic_seed(cell_seed, variant, "bootstrap"),
            )
            if not boot.empty:
                boot.insert(0, "analysis_variant", variant)
                boot.insert(0, "test_year", cohorts.test_year)
                boot.insert(0, "task_id", cohorts.task_id)
                bootstraps.append(boot)
                for interval in boot.itertuples(index=False):
                    row[f"{interval.metric}_ci_low"] = interval.ci_low
                    row[f"{interval.metric}_ci_high"] = interval.ci_high
        prediction = test[
            ["public_sample_id", "public_patient_cluster_id", "y"]
        ].copy()
        prediction.insert(0, "analysis_variant", variant)
        prediction.insert(0, "test_year", cohorts.test_year)
        prediction.insert(0, "task_id", cohorts.task_id)
        prediction["raw_probability"] = raw
        prediction["calibrated_probability"] = (
            calibrated if calibrated is not None else np.nan
        )
        predictions.append(prediction)
        rows.append(row)

    tuning = tuning.copy()
    tuning.insert(0, "test_year", cohorts.test_year)
    tuning.insert(0, "task_id", cohorts.task_id)
    return (
        rows,
        tuning,
        pd.concat(bootstraps, ignore_index=True) if bootstraps else pd.DataFrame(),
        pd.concat(predictions, ignore_index=True),
    )


def selected_tasks(task_ids: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(str(task).strip() for task in task_ids if str(task).strip()))
    unknown = set(requested) - set(TASK_IDS)
    if unknown:
        raise ValueError(f"Unknown task IDs: {sorted(unknown)}")
    requested_set = set(requested)
    return tuple(task for task in TASK_IDS if task in requested_set)
