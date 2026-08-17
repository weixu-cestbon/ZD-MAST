"""Frozen modeling primitives for the ZD-MAST primary temporal protocol."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from .metrics import expected_calibration_error


warnings.filterwarnings("ignore", message="X does not have valid feature names.*")


SEED = 20260717
N_FEATURES = 6000
LIGHTGBM_DEFAULT_SUBSAMPLE_FREQ = 0

PUBLIC_TO_LEGACY_TASK = {
    "sa_oxa": "sa_oxacillin",
    "sa_lvx": "sa_levofloxacin",
    "sa_gen": "sa_gentamicin",
    "kp_fep": "kp_cefepime",
    "kp_cro": "kp_ceftriaxone",
    "kp_caz": "kp_ceftazidime",
    "kp_cip": "kp_ciprofloxacin",
    "ec_cro": "ec_ceftriaxone",
    "ec_cip": "ec_ciprofloxacin",
    "ec_fep": "ec_cefepime",
}


@dataclass(frozen=True)
class PublicProtocolData:
    """Public-ID-only data required for one task."""

    task_id: str
    feature_matrix: np.ndarray
    development: pd.DataFrame
    test: pd.DataFrame
    folds: list[tuple[pd.DataFrame, pd.DataFrame, str]]


def stable_seed(*parts: object) -> int:
    """Return the archived deterministic seed for a run identifier."""

    token = "|".join(str(part) for part in parts).encode("utf-8")
    return SEED + int(hashlib.sha256(token).hexdigest()[:8], 16) % 1_000_000


def primary_run_id(task_id: str) -> str:
    """Construct the exact legacy run identifier used to derive seeds."""

    legacy_task = PUBLIC_TO_LEGACY_TASK[task_id]
    return (
        f"{legacy_task}__historical_S_vs_IR__lightgbm__"
        "B_post_marker_current_temporal__all_samples_workload_primary"
    )


def lightgbm_grid() -> list[dict[str, Any]]:
    """Frozen LightGBM grid from analysis version v2026.07.17.3.

    The historical grid recorded ``subsample=0.8`` without setting
    ``subsample_freq``. LightGBM therefore used its default
    ``subsample_freq=0`` and did not perform row bagging. The frozen parameter
    dictionaries are preserved byte-for-byte for result reproduction; the
    effective default is made explicit in :func:`fit_lightgbm` and audited by
    :func:`effective_lightgbm_sampling`.
    """

    return [
        {
            "num_leaves": leaves,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "min_child_samples": child,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "class_weight": weight,
        }
        for leaves in (15, 31, 63)
        for child in (20, 50)
        for weight in (None, "balanced")
    ]


def effective_lightgbm_sampling(params: dict[str, Any]) -> dict[str, Any]:
    """Describe the effective row and feature sampling settings.

    ``subsample`` only activates row sampling when ``subsample_freq`` is
    positive. Returning the effective state separately lets release manifests
    document the fitted estimator without changing archived hyperparameter
    JSON strings.
    """

    subsample = float(params.get("subsample", 1.0))
    subsample_freq = int(params.get("subsample_freq", LIGHTGBM_DEFAULT_SUBSAMPLE_FREQ))
    return {
        "subsample": subsample,
        "subsample_freq": subsample_freq,
        "row_subsampling_enabled": bool(subsample < 1.0 and subsample_freq > 0),
        "colsample_bytree": float(params.get("colsample_bytree", 1.0)),
    }


def _ordered_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "row_order" not in frame:
        raise ValueError("Frozen split is missing row_order")
    if frame["row_order"].isna().any():
        raise ValueError("Frozen split contains missing row_order")
    return frame.sort_values("row_order", kind="stable").reset_index(drop=True)


def load_public_protocol_b(release_root: Path, task_id: str) -> PublicProtocolData:
    """Load one task from the de-identified rc4 feature release.

    Cohort membership and rolling-origin folds come from frozen public split
    files. Exact dates and original identifiers are neither required nor read.
    """

    feature_root = release_root
    if not (feature_root / "zd_mast_sample_metadata_public_v1.0.0.csv").exists():
        candidates = sorted(release_root.glob("feature-release*"))
        if len(candidates) != 1:
            raise ValueError("Expected one feature-release directory")
        feature_root = candidates[0]

    matrix = np.load(
        feature_root / "zd_mast_a_sample_level_intensity6000_v1.0.0.npy",
        mmap_mode="r",
    )
    if matrix.shape[1] != N_FEATURES:
        raise ValueError(f"Expected 6000 features, found {matrix.shape[1]}")

    sample = pd.read_csv(feature_root / "zd_mast_sample_metadata_public_v1.0.0.csv")
    sample = sample[sample["site_id"].eq("ZD-MAST-A")].copy()
    labels = pd.read_parquet(feature_root / "zd_mast_ast_labels_historical_v1.0.0.parquet")
    labels = labels[
        labels["site_id"].eq("ZD-MAST-A")
        & labels["task_id"].eq(task_id)
        & labels["binary_s_vs_ir"].isin([0, 1])
    ][["public_sample_id", "binary_s_vs_ir"]].copy()
    labels["y"] = labels.pop("binary_s_vs_ir").astype(np.int8)

    groups = pd.read_parquet(feature_root / "zd_mast_patient_episode_groups_public_v1.0.0.parquet")
    if "task_id" in groups:
        groups = groups[groups["task_id"].eq(task_id)].drop(columns="task_id")
    base = labels.merge(
        sample[["public_sample_id", "feature_row"]],
        on="public_sample_id",
        how="inner",
        validate="one_to_one",
    ).merge(groups, on="public_sample_id", how="left", validate="one_to_one")

    split = pd.read_csv(feature_root / "zd_mast_split_assignments_public_v1.0.0.csv")
    split = split[
        split["analysis_id"].eq("local_temporal")
        & split["protocol"].eq("current_workflow_protocol_b")
        & split["site_id"].eq("ZD-MAST-A")
        & split["task_id"].eq(task_id)
        & split["split"].isin(["train", "test"])
    ].copy()
    if split.duplicated("public_sample_id").any():
        raise ValueError(f"Duplicate Protocol B sample assignment for {task_id}")
    assigned = split.merge(base, on="public_sample_id", how="inner", validate="one_to_one")
    if len(assigned) != len(split):
        raise ValueError(f"Protocol B references samples without a usable label/feature for {task_id}")
    development = _ordered_rows(assigned[assigned["split"].eq("train")].copy())
    test = _ordered_rows(assigned[assigned["split"].eq("test")].copy())
    if set(development["public_sample_id"]) & set(test["public_sample_id"]):
        raise ValueError(f"Development/test overlap for {task_id}")

    fold_table = pd.read_csv(
        feature_root / "zd_mast_protocol_b_rolling_origin_folds_public_v1.0.0.csv"
    )
    fold_table = fold_table[
        fold_table["analysis_id"].eq("local_temporal")
        & fold_table["protocol"].eq("current_workflow_protocol_b")
        & fold_table["task_id"].eq(task_id)
    ].copy()
    folds: list[tuple[pd.DataFrame, pd.DataFrame, str]] = []
    for fold_index in sorted(fold_table["fold_index"].unique()):
        one = fold_table[fold_table["fold_index"].eq(fold_index)].copy()
        one = one.merge(base, on="public_sample_id", how="inner", validate="many_to_one")
        if len(one) != len(fold_table[fold_table["fold_index"].eq(fold_index)]):
            raise ValueError(f"Fold {fold_index} references unusable samples for {task_id}")
        train = _ordered_rows(one[one["split"].eq("train")].copy())
        validation = _ordered_rows(one[one["split"].eq("validation")].copy())
        note_values = one["fold_note"].dropna().unique()
        note = str(note_values[0]) if len(note_values) else f"fold={fold_index}"
        folds.append((train, validation, note))
    if len(folds) < 2:
        raise ValueError(f"Fewer than two valid public folds for {task_id}")

    return PublicProtocolData(task_id, matrix, development, test, folds)


def matrix_rows(matrix: np.ndarray, frame: pd.DataFrame) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Materialize selected rows as CSR, matching the archived estimator input."""

    rows = frame["feature_row"].to_numpy(dtype=np.int64)
    if len(rows) == 0:
        return sparse.csr_matrix((0, N_FEATURES), dtype=np.float32), np.array([], dtype=np.int8)
    if rows.min() < 0 or rows.max() >= matrix.shape[0]:
        raise ValueError("feature_row out of bounds")
    dense = np.asarray(matrix[rows], dtype=np.float32)
    return sparse.csr_matrix(dense), frame["y"].to_numpy(dtype=np.int8)


def fit_lightgbm(
    params: dict[str, Any],
    x: sparse.csr_matrix,
    y: np.ndarray,
    seed: int,
    threads: int,
):
    """Fit the frozen deterministic LightGBM classifier.

    Row bagging is explicitly disabled to match the historical fit. The
    archived grid's ``subsample=0.8`` is inert under
    ``subsample_freq=0``; column subsampling remains active through
    ``colsample_bytree``.
    """

    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:  # pragma: no cover - exercised by CLI environment
        raise RuntimeError("Install the 'modeling' optional dependencies") from exc
    model_params = dict(params)
    subsample_freq = int(
        model_params.pop("subsample_freq", LIGHTGBM_DEFAULT_SUBSAMPLE_FREQ)
    )
    model = LGBMClassifier(
        objective="binary",
        random_state=seed,
        n_jobs=threads,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
        subsample_freq=subsample_freq,
        **model_params,
    )
    model.fit(x, y)
    return model


def predict(model: Any, x: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def tune_with_frozen_folds(
    data: PublicProtocolData,
    seed: int,
    threads: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Select parameters using only the frozen development folds."""

    rows: list[dict[str, Any]] = []
    for candidate_index, params in enumerate(lightgbm_grid()):
        for fold_index, (train, validation, note) in enumerate(data.folds):
            x_train, y_train = matrix_rows(data.feature_matrix, train)
            x_validation, y_validation = matrix_rows(data.feature_matrix, validation)
            model = fit_lightgbm(
                params,
                x_train,
                y_train,
                seed + candidate_index * 17 + fold_index,
                threads,
            )
            probability = predict(model, x_validation)
            rows.append(
                {
                    "candidate_index": candidate_index,
                    "params": json.dumps(params, sort_keys=True),
                    "fold_index": fold_index,
                    "fold_note": note,
                    "AUROC": roc_auc_score(y_validation, probability),
                    "AUPRC": average_precision_score(y_validation, probability),
                }
            )
    table = pd.DataFrame(rows)
    summary = table.groupby(["candidate_index", "params"], as_index=False).agg(
        folds=("fold_index", "nunique"),
        median_AUROC=("AUROC", "median"),
        median_AUPRC=("AUPRC", "median"),
    )
    summary["objective"] = summary["median_AUROC"] + 0.05 * summary["median_AUPRC"]
    best = summary.sort_values(
        ["objective", "median_AUROC", "median_AUPRC"], ascending=False
    ).iloc[0]
    return json.loads(best["params"]), table.merge(summary, on=["candidate_index", "params"])


def out_of_fold_predictions(
    data: PublicProtocolData,
    params: dict[str, Any],
    seed: int,
    threads: int,
) -> pd.DataFrame:
    """Build pooled development-only OOF predictions for calibration."""

    pieces: list[pd.DataFrame] = []
    for fold_index, (train, validation, note) in enumerate(data.folds):
        x_train, y_train = matrix_rows(data.feature_matrix, train)
        x_validation, _ = matrix_rows(data.feature_matrix, validation)
        model = fit_lightgbm(params, x_train, y_train, seed + fold_index, threads)
        piece = validation[["public_sample_id", "public_patient_cluster_id", "y"]].copy()
        piece["fold_index"] = fold_index
        piece["fold_note"] = note
        piece["raw_probability"] = predict(model, x_validation)
        pieces.append(piece)
    output = pd.concat(pieces, ignore_index=True)
    if output["public_sample_id"].duplicated().any():
        raise ValueError("OOF sample overlap detected")
    return output


def fit_platt(y: np.ndarray, probability: np.ndarray) -> LogisticRegression:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000, random_state=SEED)
    model.fit(logits, y)
    return model


def apply_platt(model: LogisticRegression, probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return model.predict_proba(logits)[:, 1]


def select_thresholds(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Select validation-derived Youden, sensitivity-90, and specificity-90 thresholds."""

    fpr, tpr, thresholds = roc_curve(y, probability)
    specificity = 1 - fpr
    finite = np.isfinite(thresholds)
    youden_index = int(np.nanargmax(np.where(finite, tpr - fpr, -np.inf)))
    sens = np.where((tpr >= 0.90) & finite)[0]
    spec = np.where((specificity >= 0.90) & finite)[0]
    sens_index = int(sens[np.argmax(specificity[sens])]) if len(sens) else None
    spec_index = int(spec[np.argmax(tpr[spec])]) if len(spec) else None
    return {
        "youden": float(thresholds[youden_index]),
        "sens90": float(thresholds[sens_index]) if sens_index is not None else float(probability.min() - 1e-9),
        "spec90": float(thresholds[spec_index]) if spec_index is not None else float(probability.max() + 1e-9),
    }


def calibration_slope_intercept(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000, random_state=SEED)
    model.fit(logits, y)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def threshold_metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (probability >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "PPV": float(ppv),
        "NPV": float(npv),
        "non_susceptible_miss_rate": float(1 - sensitivity),
        "susceptible_alert_rate": float(1 - specificity),
    }


def all_metrics(
    y: np.ndarray,
    raw: np.ndarray,
    calibrated: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, float]:
    baseline = float(y.mean())
    auprc = float(average_precision_score(y, raw))
    slope_raw, intercept_raw = calibration_slope_intercept(y, raw)
    slope_platt, intercept_platt = calibration_slope_intercept(y, calibrated)
    result = {
        "AUROC": float(roc_auc_score(y, raw)),
        "AUPRC": auprc,
        "AUPRC_baseline": baseline,
        "AUPRC_lift": auprc / baseline,
        "Brier_raw": float(brier_score_loss(y, raw)),
        "ECE_raw": expected_calibration_error(y, raw),
        "calibration_slope_raw": slope_raw,
        "calibration_intercept_raw": intercept_raw,
        "Brier_platt": float(brier_score_loss(y, calibrated)),
        "ECE_platt": expected_calibration_error(y, calibrated),
        "calibration_slope_platt": slope_platt,
        "calibration_intercept_platt": intercept_platt,
    }
    for name, threshold in thresholds.items():
        result[f"threshold_{name}"] = threshold
        for metric, value in threshold_metrics(y, calibrated, threshold).items():
            result[f"{metric}_{name}"] = value
    return result


def _bootstrap_groups(frame: pd.DataFrame) -> np.ndarray:
    groups = frame["public_patient_cluster_id"].astype("string")
    fallback = "sample:" + frame["public_sample_id"].astype(str)
    return groups.fillna(fallback).to_numpy(dtype=str)


def cluster_bootstrap(
    y: np.ndarray,
    raw: np.ndarray,
    calibrated: np.ndarray,
    groups: np.ndarray,
    thresholds: dict[str, float],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group_series = pd.Series(groups, dtype="string")
    unique_groups = group_series.unique()
    values_by_group = group_series.to_numpy()
    indices = {group: np.flatnonzero(values_by_group == group) for group in unique_groups}
    collected: dict[str, list[float]] = {}
    for _ in range(n_boot):
        selected = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        index = np.concatenate([indices[group] for group in selected])
        by = y[index]
        if np.unique(by).size < 2:
            continue
        values = all_metrics(by, raw[index], calibrated[index], thresholds)
        for metric, value in values.items():
            if not metric.startswith("threshold_") and np.isfinite(value):
                collected.setdefault(metric, []).append(float(value))
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "bootstrap_valid_n": len(values),
                "CI_low": float(np.quantile(values, 0.025)),
                "CI_high": float(np.quantile(values, 0.975)),
                "bootstrap_median": float(np.median(values)),
            }
            for metric, values in collected.items()
        ]
    )


def validate_task_data(data: PublicProtocolData) -> None:
    """Fail early on inadequate classes or split overlap."""

    for name, frame, minimum, minimum_class in (
        ("development", data.development, 100, 20),
        ("test", data.test, 30, 10),
    ):
        if len(frame) < minimum:
            raise ValueError(f"{name}: n<{minimum}")
        if frame["y"].nunique() < 2 or frame["y"].value_counts().min() < minimum_class:
            raise ValueError(f"{name}: min_class<{minimum_class}")


def run_primary_task(
    release_root: Path,
    task_id: str,
    threads: int = 4,
    n_boot: int = 1000,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reproduce one primary Protocol B task from de-identified release data."""

    data = load_public_protocol_b(release_root, task_id)
    validate_task_data(data)
    run_id = primary_run_id(task_id)
    seed = stable_seed(run_id)
    params, tuning = tune_with_frozen_folds(data, seed, threads)
    oof = out_of_fold_predictions(data, params, seed + 10000, threads)
    oof_y = oof["y"].to_numpy(dtype=np.int8)
    calibrator = fit_platt(oof_y, oof["raw_probability"].to_numpy(dtype=float))
    oof["platt_probability"] = apply_platt(calibrator, oof["raw_probability"].to_numpy(dtype=float))
    thresholds = select_thresholds(oof_y, oof["platt_probability"].to_numpy(dtype=float))

    x_development, y_development = matrix_rows(data.feature_matrix, data.development)
    x_test, y_test = matrix_rows(data.feature_matrix, data.test)
    model = fit_lightgbm(params, x_development, y_development, seed + 20000, threads)
    raw = predict(model, x_test)
    calibrated = apply_platt(calibrator, raw)
    result: dict[str, Any] = {
        "run_id": run_id,
        "task_id": task_id,
        "analysis_id": "local_temporal",
        "protocol": "current_workflow_protocol_b",
        "analysis_version": "v2026.07.17.3",
        "model": "lightgbm",
        "endpoint": "historical_S_vs_IR",
        "status": "ok",
        "n_development": len(data.development),
        "development_positive_n": int(y_development.sum()),
        "development_negative_n": int((y_development == 0).sum()),
        "n_test": len(data.test),
        "test_positive_n": int(y_test.sum()),
        "test_negative_n": int((y_test == 0).sum()),
        "test_episode_n": int(data.test["public_episode_id"].nunique()),
        "rolling_origin_fold_n": len(data.folds),
        "oof_n": len(oof),
        "best_hyperparameters": json.dumps(params, sort_keys=True),
        **all_metrics(y_test, raw, calibrated, thresholds),
    }
    bootstrap = cluster_bootstrap(
        y_test,
        raw,
        calibrated,
        _bootstrap_groups(data.test),
        thresholds,
        n_boot,
        seed + 30000,
    )
    for row in bootstrap.itertuples(index=False):
        result[f"{row.metric}_CI_low"] = row.CI_low
        result[f"{row.metric}_CI_high"] = row.CI_high
    predictions = data.test[["public_sample_id", "y"]].copy()
    predictions["raw_probability"] = raw
    predictions["platt_probability"] = calibrated
    predictions.insert(0, "task_id", task_id)
    tuning.insert(0, "task_id", task_id)
    bootstrap.insert(0, "task_id", task_id)
    oof.insert(0, "task_id", task_id)
    return result, tuning, bootstrap, predictions


def _episode_first_protocol(data: PublicProtocolData) -> PublicProtocolData:
    """Restrict a frozen public protocol to patient-species 30-day episode first rows."""

    def keep(frame: pd.DataFrame) -> pd.DataFrame:
        selected = frame[
            frame["public_patient_cluster_id"].notna()
            & frame["episode_first_sample_flag"].fillna(False).astype(bool)
        ].copy()
        return _ordered_rows(selected)

    folds = [(keep(train), keep(validation), note) for train, validation, note in data.folds]
    folds = [
        (train, validation, note)
        for train, validation, note in folds
        if len(train) >= 50
        and len(validation) >= 20
        and train["y"].nunique() == 2
        and validation["y"].nunique() == 2
        and train["y"].value_counts().min() >= 10
        and validation["y"].value_counts().min() >= 5
    ]
    return PublicProtocolData(
        task_id=data.task_id,
        feature_matrix=data.feature_matrix,
        development=keep(data.development),
        test=keep(data.test),
        folds=folds,
    )


def _sensitivity_result_base(
    data: PublicProtocolData,
    run_id: str,
    analysis_variant: str,
    params: dict[str, Any],
    hyperparameter_source: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "task_id": data.task_id,
        "analysis_id": "patient_episode_sensitivity",
        "protocol": "current_workflow_protocol_b",
        "analysis_version": "v2026.07.17.3",
        "model": "lightgbm",
        "endpoint": "historical_S_vs_IR",
        "analysis_variant": analysis_variant,
        "development_definition": "2025-07-01 through 2026-02-28",
        "test_definition": "2026-03-01 through 2026-06-09",
        "n_development": len(data.development),
        "development_positive_n": int(data.development["y"].sum()),
        "development_negative_n": int(data.development["y"].eq(0).sum()),
        "n_test": len(data.test),
        "test_positive_n": int(data.test["y"].sum()),
        "test_negative_n": int(data.test["y"].eq(0).sum()),
        "test_patient_n": int(data.test["public_patient_cluster_id"].nunique()),
        "test_episode_n": int(data.test["public_episode_id"].nunique()),
        "rolling_origin_fold_n": len(data.folds),
        "hyperparameter_source": hyperparameter_source,
        "best_hyperparameters": json.dumps(params, sort_keys=True),
    }


def run_patient_episode_sensitivity_task(
    release_root: Path,
    task_id: str,
    primary_params: dict[str, Any],
    threads: int = 4,
    n_boot: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce patient-disjoint and episode-first Protocol B sensitivities.

    Hyperparameters are fixed from the all-sample primary analysis. Calibration,
    thresholds, and all model fitting use development-only frozen folds.
    """

    data = load_public_protocol_b(release_root, task_id)
    validate_task_data(data)
    primary_id = primary_run_id(task_id)
    primary_seed = stable_seed(primary_id)
    primary_oof = out_of_fold_predictions(data, primary_params, primary_seed + 10000, threads)
    primary_oof_y = primary_oof["y"].to_numpy(dtype=np.int8)
    primary_calibrator = fit_platt(
        primary_oof_y,
        primary_oof["raw_probability"].to_numpy(dtype=float),
    )
    primary_oof["platt_probability"] = apply_platt(
        primary_calibrator,
        primary_oof["raw_probability"].to_numpy(dtype=float),
    )
    primary_thresholds = select_thresholds(
        primary_oof_y,
        primary_oof["platt_probability"].to_numpy(dtype=float),
    )
    x_development, y_development = matrix_rows(data.feature_matrix, data.development)
    x_test, y_test = matrix_rows(data.feature_matrix, data.test)
    primary_model = fit_lightgbm(
        primary_params,
        x_development,
        y_development,
        primary_seed + 20000,
        threads,
    )
    primary_raw = predict(primary_model, x_test)
    primary_calibrated = apply_platt(primary_calibrator, primary_raw)

    development_patients = set(data.development["public_patient_cluster_id"].dropna().astype(str))
    test_patient = data.test["public_patient_cluster_id"].astype("string")
    disjoint_mask = test_patient.notna() & ~test_patient.astype(str).isin(development_patients)
    disjoint = _ordered_rows(data.test.loc[disjoint_mask].copy())
    disjoint_index = np.flatnonzero(disjoint_mask.to_numpy())
    disjoint_id = primary_id + "__patient_disjoint_test"
    disjoint_base = _sensitivity_result_base(
        PublicProtocolData(task_id, data.feature_matrix, data.development, disjoint, data.folds),
        disjoint_id,
        "patient_disjoint_test",
        primary_params,
        "rolling_origin_selection",
    )
    if len(disjoint) < 30 or disjoint["y"].nunique() < 2 or disjoint["y"].value_counts().min() < 10:
        raise ValueError(f"Patient-disjoint test below adequacy threshold for {task_id}")
    disjoint_y = y_test[disjoint_index]
    disjoint_raw = primary_raw[disjoint_index]
    disjoint_calibrated = primary_calibrated[disjoint_index]
    disjoint_result = {
        **disjoint_base,
        "status": "ok",
        "oof_n": len(primary_oof),
        "oof_positive_n": int(primary_oof_y.sum()),
        "oof_negative_n": int((primary_oof_y == 0).sum()),
        **all_metrics(disjoint_y, disjoint_raw, disjoint_calibrated, primary_thresholds),
    }
    disjoint_bootstrap = cluster_bootstrap(
        disjoint_y,
        disjoint_raw,
        disjoint_calibrated,
        disjoint["public_patient_cluster_id"].astype(str).to_numpy(),
        primary_thresholds,
        n_boot,
        stable_seed(disjoint_id),
    )
    for row in disjoint_bootstrap.itertuples(index=False):
        disjoint_result[f"{row.metric}_CI_low"] = row.CI_low
        disjoint_result[f"{row.metric}_CI_high"] = row.CI_high
    disjoint_bootstrap.insert(0, "run_id", disjoint_id)

    episode = _episode_first_protocol(data)
    validate_task_data(episode)
    if len(episode.folds) < 2:
        raise ValueError(f"Fewer than two episode-first folds for {task_id}")
    legacy_task = PUBLIC_TO_LEGACY_TASK[task_id]
    episode_id = (
        f"{legacy_task}__historical_S_vs_IR__lightgbm__"
        "B_post_marker_current_temporal__patient_species_30d_episode_first"
    )
    episode_seed = stable_seed(episode_id)
    episode_oof = out_of_fold_predictions(episode, primary_params, episode_seed + 10000, threads)
    episode_oof_y = episode_oof["y"].to_numpy(dtype=np.int8)
    episode_calibrator = fit_platt(
        episode_oof_y,
        episode_oof["raw_probability"].to_numpy(dtype=float),
    )
    episode_oof["platt_probability"] = apply_platt(
        episode_calibrator,
        episode_oof["raw_probability"].to_numpy(dtype=float),
    )
    episode_thresholds = select_thresholds(
        episode_oof_y,
        episode_oof["platt_probability"].to_numpy(dtype=float),
    )
    x_episode_development, y_episode_development = matrix_rows(
        episode.feature_matrix,
        episode.development,
    )
    x_episode_test, y_episode_test = matrix_rows(episode.feature_matrix, episode.test)
    episode_model = fit_lightgbm(
        primary_params,
        x_episode_development,
        y_episode_development,
        episode_seed + 20000,
        threads,
    )
    episode_raw = predict(episode_model, x_episode_test)
    episode_calibrated = apply_platt(episode_calibrator, episode_raw)
    episode_result = {
        **_sensitivity_result_base(
            episode,
            episode_id,
            "patient_species_30d_episode_first",
            primary_params,
            "fixed_from_all_samples_same_task_protocol_label",
        ),
        "status": "ok",
        "oof_n": len(episode_oof),
        "oof_positive_n": int(episode_oof_y.sum()),
        "oof_negative_n": int((episode_oof_y == 0).sum()),
        **all_metrics(y_episode_test, episode_raw, episode_calibrated, episode_thresholds),
    }
    episode_bootstrap = cluster_bootstrap(
        y_episode_test,
        episode_raw,
        episode_calibrated,
        _bootstrap_groups(episode.test),
        episode_thresholds,
        n_boot,
        episode_seed + 30000,
    )
    for row in episode_bootstrap.itertuples(index=False):
        episode_result[f"{row.metric}_CI_low"] = row.CI_low
        episode_result[f"{row.metric}_CI_high"] = row.CI_high
    episode_bootstrap.insert(0, "run_id", episode_id)
    return pd.DataFrame([disjoint_result, episode_result]), pd.concat(
        [disjoint_bootstrap, episode_bootstrap],
        ignore_index=True,
    )


def compare_sensitivities_to_frozen(
    reproduced: pd.DataFrame,
    frozen_path: Path,
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Compare public sensitivity reruns with frozen Protocol B LightGBM rows."""

    frozen = pd.read_csv(frozen_path)
    if "endpoint" in frozen.columns:
        endpoint = frozen["endpoint"]
        protocol = "current_workflow_protocol_b"
        public_schema = True
    elif "label_type" in frozen.columns:
        endpoint = frozen["label_type"]
        protocol = "B_post_marker_current_temporal"
        public_schema = False
    else:
        raise ValueError("Frozen sensitivity table lacks endpoint or label_type")
    frozen = frozen[
        endpoint.eq("historical_S_vs_IR")
        & frozen["model"].eq("lightgbm")
        & frozen["protocol"].eq(protocol)
        & frozen["analysis_variant"].isin(
            ["patient_disjoint_test", "patient_species_30d_episode_first"]
        )
    ].copy()
    if not public_schema:
        frozen["task_id"] = frozen["task_id"].map(
            {legacy: public for public, legacy in PUBLIC_TO_LEGACY_TASK.items()}
        )
    keys = ["task_id", "analysis_variant"]
    if set(map(tuple, frozen[keys].to_numpy())) != set(map(tuple, reproduced[keys].to_numpy())):
        raise ValueError("Frozen/reproduced sensitivity task sets differ")
    joined = frozen.merge(reproduced, on=keys, suffixes=("_frozen", "_reproduced"), validate="one_to_one")
    fields: Iterable[str] = (
        "n_development",
        "development_positive_n",
        "development_negative_n",
        "n_test",
        "test_positive_n",
        "test_negative_n",
        "test_patient_n",
        "test_episode_n",
        "rolling_origin_fold_n",
        "oof_n",
        "AUROC",
        "AUPRC",
        "AUPRC_baseline",
        "AUPRC_lift",
        "Brier_platt",
        "ECE_platt",
        "sensitivity_sens90",
        "specificity_sens90",
        "non_susceptible_miss_rate_sens90",
    )
    rows: list[dict[str, Any]] = []
    count_fields = {
        "n_development",
        "development_positive_n",
        "development_negative_n",
        "n_test",
        "test_positive_n",
        "test_negative_n",
        "test_patient_n",
        "test_episode_n",
        "rolling_origin_fold_n",
        "oof_n",
    }
    for row in joined.itertuples(index=False):
        for field in fields:
            frozen_value = getattr(row, f"{field}_frozen")
            reproduced_value = getattr(row, f"{field}_reproduced")
            difference = abs(float(reproduced_value) - float(frozen_value))
            field_tolerance = 0.0 if field in count_fields else tolerance
            corrected_frozen_metadata = (
                row.analysis_variant == "patient_disjoint_test"
                and field == "test_episode_n"
                and difference > field_tolerance
            )
            rows.append(
                {
                    "task_id": row.task_id,
                    "analysis_variant": row.analysis_variant,
                    "field": field,
                    "frozen_value": frozen_value,
                    "reproduced_value": reproduced_value,
                    "absolute_difference": difference,
                    "tolerance": field_tolerance,
                    "status": (
                        "CORRECTED_FROZEN_METADATA"
                        if corrected_frozen_metadata
                        else "PASS"
                        if difference <= field_tolerance
                        else "FAIL"
                    ),
                    "note": (
                        "Frozen checkpoint retained the full-test episode count; the reproduced value is recomputed after the patient-disjoint filter."
                        if corrected_frozen_metadata
                        else ""
                    ),
                }
            )
    return pd.DataFrame(rows)


def compare_to_frozen(
    reproduced: pd.DataFrame,
    frozen_path: Path,
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Compare manuscript-facing fields with the frozen aggregate table."""

    frozen = pd.read_csv(frozen_path)
    if set(frozen["task_id"]) != set(reproduced["task_id"]):
        raise ValueError(
            "Frozen/reproduced task sets differ: "
            f"frozen={sorted(frozen['task_id'])}, reproduced={sorted(reproduced['task_id'])}"
        )
    joined = frozen.merge(reproduced, on="task_id", suffixes=("_frozen", "_reproduced"), validate="one_to_one")
    fields: Iterable[str] = (
        "n_development",
        "development_positive_n",
        "development_negative_n",
        "n_test",
        "test_positive_n",
        "test_negative_n",
        "test_episode_n",
        "AUROC",
        "AUROC_CI_low",
        "AUROC_CI_high",
        "AUPRC",
        "AUPRC_baseline",
        "AUPRC_lift",
        "Brier_platt",
        "ECE_platt",
        "sensitivity_sens90",
        "specificity_sens90",
        "non_susceptible_miss_rate_sens90",
    )
    rows: list[dict[str, Any]] = []
    for row in joined.itertuples(index=False):
        task_id = getattr(row, "task_id")
        for field in fields:
            frozen_value = getattr(row, f"{field}_frozen")
            reproduced_value = getattr(row, f"{field}_reproduced")
            if field.startswith("n_") or field.endswith("_n"):
                difference = float(reproduced_value) - float(frozen_value)
                passed = difference == 0
            else:
                difference = abs(float(reproduced_value) - float(frozen_value))
                passed = difference <= tolerance
            rows.append(
                {
                    "task_id": task_id,
                    "field": field,
                    "frozen_value": frozen_value,
                    "reproduced_value": reproduced_value,
                    "absolute_difference": abs(difference),
                    "tolerance": 0.0 if field.startswith("n_") or field.endswith("_n") else tolerance,
                    "status": "PASS" if passed else "FAIL",
                }
            )
    return pd.DataFrame(rows)
