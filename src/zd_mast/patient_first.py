"""Strict one-record-per-patient sensitivity for frozen Protocol B."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .modeling import (
    PublicProtocolData,
    _bootstrap_groups,
    all_metrics,
    apply_platt,
    cluster_bootstrap,
    fit_lightgbm,
    fit_platt,
    load_public_protocol_b,
    matrix_rows,
    out_of_fold_predictions,
    predict,
    primary_run_id,
    select_thresholds,
    stable_seed,
)


ANALYSIS_ID = "patient_first_sensitivity"
VARIANT = "strict_one_record_per_patient"


@dataclass(frozen=True)
class PatientFirstAudit:
    development_input_n: int
    development_output_n: int
    test_input_n: int
    test_removed_seen_patient_n: int
    test_output_n: int
    fold_input_n: int
    fold_output_n: int


def first_record_per_patient(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the earliest ordered row for each non-missing patient cluster."""

    required = {"public_patient_cluster_id", "row_order"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Patient-first frame missing columns: {sorted(missing)}")
    selected = frame.loc[frame["public_patient_cluster_id"].notna()].copy()
    selected = selected.sort_values(
        ["row_order", "public_sample_id"], kind="stable"
    ).drop_duplicates("public_patient_cluster_id", keep="first")
    return selected.sort_values("row_order", kind="stable").reset_index(drop=True)


def build_patient_first_protocol(
    data: PublicProtocolData,
) -> tuple[PublicProtocolData, PatientFirstAudit, pd.DataFrame]:
    """Construct a patient-first, cross-period patient-disjoint protocol.

    Development is reduced to the first ordered sample per patient. Test rows
    from any development patient are then removed, after which the first test
    row per remaining patient is retained. Each development fold applies the
    same rule independently, with validation patients purged against training.
    """

    development = first_record_per_patient(data.development)
    seen = set(development["public_patient_cluster_id"].astype(str))
    test_candidates = data.test.loc[
        data.test["public_patient_cluster_id"].notna()
        & ~data.test["public_patient_cluster_id"].astype(str).isin(seen)
    ].copy()
    removed_seen = len(data.test) - len(test_candidates)
    test = first_record_per_patient(test_candidates)

    folds: list[tuple[pd.DataFrame, pd.DataFrame, str]] = []
    audit_rows: list[dict[str, Any]] = []
    fold_input_n = 0
    fold_output_n = 0
    for fold_index, (train_raw, validation_raw, note) in enumerate(data.folds):
        fold_input_n += len(train_raw) + len(validation_raw)
        train = first_record_per_patient(train_raw)
        train_patients = set(train["public_patient_cluster_id"].astype(str))
        validation_candidates = validation_raw.loc[
            validation_raw["public_patient_cluster_id"].notna()
            & ~validation_raw["public_patient_cluster_id"].astype(str).isin(
                train_patients
            )
        ].copy()
        validation = first_record_per_patient(validation_candidates)
        fold_output_n += len(train) + len(validation)
        train_class = train["y"].value_counts()
        validation_class = validation["y"].value_counts()
        adequate = (
            len(train) >= 50
            and train_class.size == 2
            and int(train_class.min()) >= 10
            and len(validation) >= 20
            and validation_class.size == 2
            and int(validation_class.min()) >= 5
        )
        audit_rows.append(
            {
                "task_id": data.task_id,
                "fold_index": fold_index,
                "fold_note": note,
                "train_input_n": len(train_raw),
                "train_patient_first_n": len(train),
                "validation_input_n": len(validation_raw),
                "validation_removed_seen_patient_n": len(validation_raw)
                - len(validation_candidates),
                "validation_patient_first_n": len(validation),
                "status": "adequate" if adequate else "insufficient",
            }
        )
        if adequate:
            folds.append((train, validation, note))

    audit = PatientFirstAudit(
        development_input_n=len(data.development),
        development_output_n=len(development),
        test_input_n=len(data.test),
        test_removed_seen_patient_n=removed_seen,
        test_output_n=len(test),
        fold_input_n=fold_input_n,
        fold_output_n=fold_output_n,
    )
    return (
        PublicProtocolData(
            task_id=data.task_id,
            feature_matrix=data.feature_matrix,
            development=development,
            test=test,
            folds=folds,
        ),
        audit,
        pd.DataFrame(audit_rows),
    )


def run_patient_first_task(
    release_root: Path,
    task_id: str,
    fixed_params: dict[str, Any],
    *,
    threads: int,
    n_boot: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit and evaluate one strict patient-first Protocol B sensitivity."""

    source = load_public_protocol_b(release_root, task_id)
    data, audit, fold_audit = build_patient_first_protocol(source)
    if len(data.folds) < 2:
        raise ValueError(f"{task_id}: fewer than two adequate patient-first folds")
    if len(data.test) < 30 or data.test["y"].value_counts().min() < 10:
        raise ValueError(f"{task_id}: patient-first test support is insufficient")

    seed = stable_seed(primary_run_id(task_id), ANALYSIS_ID)
    oof = out_of_fold_predictions(data, fixed_params, seed + 10_000, threads)
    oof_y = oof["y"].to_numpy(dtype=np.int8)
    calibrator = fit_platt(oof_y, oof["raw_probability"].to_numpy(dtype=float))
    oof["platt_probability"] = apply_platt(
        calibrator, oof["raw_probability"].to_numpy(dtype=float)
    )
    thresholds = select_thresholds(
        oof_y, oof["platt_probability"].to_numpy(dtype=float)
    )

    x_development, y_development = matrix_rows(data.feature_matrix, data.development)
    x_test, y_test = matrix_rows(data.feature_matrix, data.test)
    model = fit_lightgbm(
        fixed_params, x_development, y_development, seed + 20_000, threads
    )
    raw = predict(model, x_test)
    calibrated = apply_platt(calibrator, raw)
    result: dict[str, Any] = {
        "analysis_id": ANALYSIS_ID,
        "analysis_variant": VARIANT,
        "task_id": task_id,
        "model": "lightgbm",
        "endpoint": "historical_S_vs_IR",
        "feature_representation": "intensity6000",
        "protocol": "current_workflow_protocol_b",
        "status": "ok",
        "n_development": len(data.development),
        "development_positive_n": int(y_development.sum()),
        "development_negative_n": int(y_development.size - y_development.sum()),
        "n_test": len(data.test),
        "test_positive_n": int(y_test.sum()),
        "test_negative_n": int(y_test.size - y_test.sum()),
        "test_patient_n": int(data.test["public_patient_cluster_id"].nunique()),
        "valid_fold_n": len(data.folds),
        "oof_n": len(oof),
        "best_hyperparameters": json.dumps(fixed_params, sort_keys=True),
        "hyperparameter_source": "frozen_current_workflow_protocol_b_same_task",
        "development_input_n": audit.development_input_n,
        "development_patient_first_n": audit.development_output_n,
        "test_input_n": audit.test_input_n,
        "test_removed_seen_patient_n": audit.test_removed_seen_patient_n,
        "test_patient_first_n": audit.test_output_n,
        **all_metrics(y_test, raw, calibrated, thresholds),
    }
    bootstrap = cluster_bootstrap(
        y_test,
        raw,
        calibrated,
        _bootstrap_groups(data.test),
        thresholds,
        n_boot,
        seed + 30_000,
    )
    for row in bootstrap.itertuples(index=False):
        result[f"{row.metric}_CI_low"] = row.CI_low
        result[f"{row.metric}_CI_high"] = row.CI_high
    prediction = data.test[
        ["public_sample_id", "public_patient_cluster_id", "y"]
    ].copy()
    prediction.insert(0, "task_id", task_id)
    prediction["raw_probability"] = raw
    prediction["platt_probability"] = calibrated
    bootstrap.insert(0, "task_id", task_id)
    return result, prediction, bootstrap, fold_audit
