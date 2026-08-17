"""Current-contract logistic-regression sensitivity analyses for ZD-MAST.

This module deliberately contains no cohort-construction logic. Callers must
obtain development, validation, and evaluation rows from the frozen annual,
training-history, or cross-platform loaders. The fitting API accepts only the
development rows and development-only folds, which makes target-domain label
use impossible at the estimator boundary.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .cross_platform import (
    _bootstrap_groups,
    _cluster_bootstrap_intervals,
    classify_support,
    deterministic_seed,
    probability_metrics,
)
from .modeling import matrix_rows


ANALYSIS_ID = "current_contract_logistic_sensitivity"
MODEL_ID = "logistic_regression"
DEFAULT_SEED = 20260817
C_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
CLASS_WEIGHT_GRID: tuple[str | None, ...] = (None, "balanced")
RAW_METRIC_NAMES: tuple[str, ...] = (
    "raw_auroc",
    "raw_auprc",
    "auprc_baseline",
    "raw_auprc_lift",
    "raw_brier",
    "raw_ece",
    "raw_calibration_slope",
    "raw_calibration_intercept",
)


@dataclass(frozen=True)
class LogisticFit:
    """One source/development-only fitted logistic baseline."""

    task_id: str
    model: Pipeline
    best_parameters: Mapping[str, Any]
    tuning: pd.DataFrame
    oof_predictions: pd.DataFrame
    seed: int
    development_membership_sha256: str
    fold_membership_sha256: str
    final_fit_converged: bool
    final_fit_iterations: int


def logistic_grid() -> list[dict[str, object]]:
    """Return the prespecified eight-candidate L2 logistic grid."""

    return [
        {
            "C": c_value,
            "class_weight": class_weight,
            "penalty": "l2",
            "solver": "liblinear",
            "max_iter": 5000,
            "tol": 1e-4,
        }
        for c_value in C_GRID
        for class_weight in CLASS_WEIGHT_GRID
    ]


def _canonical_parameter_json(parameters: Mapping[str, object]) -> str:
    return json.dumps(dict(parameters), sort_keys=True, separators=(",", ":"))


def membership_sha256(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] = ("public_sample_id",),
) -> str:
    """Hash ordered public membership fields without exposing private values."""

    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Membership hash columns are missing: {sorted(missing)}")
    canonical = frame.loc[:, list(columns)].astype("string").fillna("<NA>")
    payload = "\n".join("\t".join(row) for row in canonical.itertuples(index=False, name=None))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fold_membership_sha256(
    folds: Sequence[tuple[pd.DataFrame, pd.DataFrame, str]],
) -> str:
    """Hash exact ordered train/validation memberships and fold notes."""

    digest = hashlib.sha256()
    for fold_index, (train, validation, note) in enumerate(folds):
        digest.update(f"fold={fold_index}\tnote={note}\n".encode("utf-8"))
        for split_name, frame in (("train", train), ("validation", validation)):
            digest.update(f"split={split_name}\n".encode("utf-8"))
            digest.update(membership_sha256(frame).encode("ascii"))
            digest.update(b"\n")
    return digest.hexdigest()


def validate_development_contract(
    development: pd.DataFrame,
    folds: Sequence[tuple[pd.DataFrame, pd.DataFrame, str]],
    *,
    evaluation_frames: Sequence[pd.DataFrame] = (),
) -> None:
    """Reject split leakage or malformed development-only folds.

    ``evaluation_frames`` are inspected only for membership overlap. Their
    labels are never read here and never enter fitting or tuning.
    """

    required = {"public_sample_id", "feature_row", "y"}
    missing = required.difference(development.columns)
    if missing:
        raise ValueError(f"Development rows are missing columns: {sorted(missing)}")
    if development.empty:
        raise ValueError("Development cohort is empty")
    if development["public_sample_id"].astype(str).duplicated().any():
        raise ValueError("Development cohort contains duplicate public_sample_id values")
    if not development["y"].isin([0, 1]).all() or development["y"].nunique() != 2:
        raise ValueError("Development labels must contain both binary classes")
    if len(folds) < 2:
        raise ValueError("At least two frozen development folds are required")

    development_ids = set(development["public_sample_id"].astype(str))
    validation_ids_seen: set[str] = set()
    for fold_index, (train, validation, _note) in enumerate(folds):
        for split_name, frame in (("train", train), ("validation", validation)):
            split_missing = required.difference(frame.columns)
            if split_missing:
                raise ValueError(
                    f"Fold {fold_index} {split_name} is missing columns: {sorted(split_missing)}"
                )
            if frame.empty or frame["y"].nunique() != 2:
                raise ValueError(f"Fold {fold_index} {split_name} lacks two-class support")
            split_ids = set(frame["public_sample_id"].astype(str))
            if not split_ids.issubset(development_ids):
                raise ValueError(f"Fold {fold_index} {split_name} includes non-development rows")
        train_ids = set(train["public_sample_id"].astype(str))
        validation_ids = set(validation["public_sample_id"].astype(str))
        if train_ids & validation_ids:
            raise ValueError(f"Fold {fold_index} has train/validation sample overlap")
        if validation_ids_seen & validation_ids:
            raise ValueError("A validation sample appears in more than one frozen fold")
        validation_ids_seen.update(validation_ids)

    for frame in evaluation_frames:
        if "public_sample_id" not in frame:
            raise ValueError("Evaluation frame lacks public_sample_id")
        overlap = development_ids & set(frame["public_sample_id"].astype(str))
        if overlap:
            raise ValueError(f"Development/evaluation sample overlap detected: n={len(overlap)}")


def fit_logistic_model(
    x: sparse.csr_matrix,
    y: np.ndarray,
    parameters: Mapping[str, object],
    *,
    seed: int,
) -> tuple[Pipeline, dict[str, object]]:
    """Fit an L2 logistic pipeline with sparse-safe, fold-local scaling."""

    if not sparse.isspmatrix_csr(x):
        x = sparse.csr_matrix(x)
    y_array = np.asarray(y, dtype=np.int8)
    if x.shape[0] != y_array.size:
        raise ValueError("Feature and label row counts differ")
    if np.unique(y_array).size != 2:
        raise ValueError("Logistic fitting requires both classes")
    candidate = dict(parameters)
    if candidate.get("penalty") != "l2":
        raise ValueError("Only L2 logistic regression is allowed")
    if candidate.get("solver") != "liblinear":
        raise ValueError("The frozen sensitivity solver is liblinear")
    pipeline = Pipeline(
        steps=[
            ("scale", StandardScaler(with_mean=False)),
            (
                "logistic",
                LogisticRegression(
                    C=float(candidate["C"]),
                    class_weight=candidate["class_weight"],
                    penalty="l2",
                    solver="liblinear",
                    max_iter=int(candidate["max_iter"]),
                    tol=float(candidate["tol"]),
                    random_state=int(seed),
                ),
            ),
        ]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(x, y_array)
    estimator = pipeline.named_steps["logistic"]
    convergence_warnings = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
    audit = {
        "converged": not convergence_warnings,
        "convergence_warning": " | ".join(str(item.message) for item in convergence_warnings),
        "n_iter": int(np.max(estimator.n_iter_)),
        "standardization": "StandardScaler(with_mean=False)",
        "sparse_centering_disabled": True,
    }
    return pipeline, audit


def predict_probability(model: Pipeline, x: sparse.csr_matrix) -> np.ndarray:
    probability = np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Logistic probabilities must be finite values in [0, 1]")
    return probability


def tune_logistic_with_frozen_folds(
    matrix: np.ndarray,
    folds: Sequence[tuple[pd.DataFrame, pd.DataFrame, str]],
    *,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Tune only within frozen development folds.

    The selection objective matches the existing LightGBM contract:
    median AUROC + 0.05 * median AUPRC. No evaluation frame is accepted.
    """

    candidates = logistic_grid()
    rows: list[dict[str, object]] = []
    for fold_index, (train, validation, note) in enumerate(folds):
        x_train, y_train = matrix_rows(matrix, train)
        x_validation, y_validation = matrix_rows(matrix, validation)
        for candidate_index, parameters in enumerate(candidates):
            model, fit_audit = fit_logistic_model(
                x_train,
                y_train,
                parameters,
                seed=deterministic_seed(seed, "tuning", candidate_index, fold_index),
            )
            probability = predict_probability(model, x_validation)
            rows.append(
                {
                    "candidate_index": candidate_index,
                    "parameters": _canonical_parameter_json(parameters),
                    "fold_index": fold_index,
                    "fold_note": note,
                    "n_train": int(len(y_train)),
                    "n_validation": int(len(y_validation)),
                    "validation_positive_n": int(y_validation.sum()),
                    "validation_negative_n": int(y_validation.size - y_validation.sum()),
                    "AUROC": float(roc_auc_score(y_validation, probability)),
                    "AUPRC": float(average_precision_score(y_validation, probability)),
                    **fit_audit,
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No logistic tuning rows were generated")
    summary = table.groupby(["candidate_index", "parameters"], as_index=False).agg(
        fold_n=("fold_index", "nunique"),
        median_AUROC=("AUROC", "median"),
        median_AUPRC=("AUPRC", "median"),
        all_folds_converged=("converged", "all"),
    )
    summary["selection_objective"] = (
        summary["median_AUROC"] + 0.05 * summary["median_AUPRC"]
    )
    best = summary.sort_values(
        ["selection_objective", "median_AUROC", "median_AUPRC", "candidate_index"],
        ascending=[False, False, False, True],
        kind="stable",
    ).iloc[0]
    best_index = int(best["candidate_index"])
    merged = table.merge(summary, on=["candidate_index", "parameters"], validate="many_to_one")
    merged["selected_candidate"] = merged["candidate_index"].eq(best_index)
    return dict(candidates[best_index]), merged


def out_of_fold_logistic_predictions(
    matrix: np.ndarray,
    folds: Sequence[tuple[pd.DataFrame, pd.DataFrame, str]],
    parameters: Mapping[str, object],
    *,
    seed: int,
) -> pd.DataFrame:
    """Generate audit-only development OOF predictions for the selected model."""

    pieces: list[pd.DataFrame] = []
    for fold_index, (train, validation, note) in enumerate(folds):
        x_train, y_train = matrix_rows(matrix, train)
        x_validation, _ = matrix_rows(matrix, validation)
        model, fit_audit = fit_logistic_model(
            x_train,
            y_train,
            parameters,
            seed=deterministic_seed(seed, "oof", fold_index),
        )
        columns = ["public_sample_id", "y"]
        if "public_patient_cluster_id" in validation:
            columns.insert(1, "public_patient_cluster_id")
        piece = validation[columns].copy()
        piece["fold_index"] = fold_index
        piece["fold_note"] = note
        piece["raw_probability"] = predict_probability(model, x_validation)
        piece["fit_converged"] = bool(fit_audit["converged"])
        pieces.append(piece)
    output = pd.concat(pieces, ignore_index=True)
    if output["public_sample_id"].astype(str).duplicated().any():
        raise ValueError("OOF validation sample overlap detected")
    return output


def fit_development_logistic(
    matrix: np.ndarray,
    task_id: str,
    development: pd.DataFrame,
    folds: Sequence[tuple[pd.DataFrame, pd.DataFrame, str]],
    *,
    seed: int = DEFAULT_SEED,
    evaluation_frames_for_overlap_audit: Sequence[pd.DataFrame] = (),
) -> LogisticFit:
    """Tune and fit one baseline without accepting target labels."""

    validate_development_contract(
        development,
        folds,
        evaluation_frames=evaluation_frames_for_overlap_audit,
    )
    child_seed = deterministic_seed(seed, ANALYSIS_ID, task_id)
    best, tuning = tune_logistic_with_frozen_folds(matrix, folds, seed=child_seed)
    oof = out_of_fold_logistic_predictions(
        matrix,
        folds,
        best,
        seed=child_seed + 10_000,
    )
    x_development, y_development = matrix_rows(matrix, development)
    model, final_audit = fit_logistic_model(
        x_development,
        y_development,
        best,
        seed=child_seed + 20_000,
    )
    return LogisticFit(
        task_id=task_id,
        model=model,
        best_parameters=best,
        tuning=tuning,
        oof_predictions=oof,
        seed=child_seed,
        development_membership_sha256=membership_sha256(development),
        fold_membership_sha256=fold_membership_sha256(folds),
        final_fit_converged=bool(final_audit["converged"]),
        final_fit_iterations=int(final_audit["n_iter"]),
    )


def raw_probability_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Return only the prespecified raw probability metrics."""

    values = probability_metrics(y, probability, None, {})
    return {name: float(values[name]) for name in RAW_METRIC_NAMES}


def evaluate_logistic_fit(
    fitted: LogisticFit,
    matrix: np.ndarray,
    frame: pd.DataFrame,
    *,
    protocol_family: str,
    cohort_id: str,
    site_id: str,
    feature_representation: str,
    bootstrap_count: int,
    seed: int,
    extra_fields: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Evaluate a fitted model; this function never refits or recalibrates it."""

    required = {"public_sample_id", "feature_row", "y"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Evaluation frame is missing columns: {sorted(missing)}")
    x_test, y_test = matrix_rows(matrix, frame)
    probability = predict_probability(fitted.model, x_test)
    positive_n = int(y_test.sum())
    negative_n = int(y_test.size - positive_n)
    support = classify_support(len(y_test), positive_n, negative_n)
    groups, group_source = _bootstrap_groups(frame)
    metrics = raw_probability_metrics(y_test, probability)
    row: dict[str, object] = {
        "analysis_id": ANALYSIS_ID,
        "protocol_family": protocol_family,
        "task_id": fitted.task_id,
        "model": MODEL_ID,
        "endpoint": "historical_S_vs_IR",
        "feature_representation": feature_representation,
        "cohort_id": cohort_id,
        "site_id": site_id,
        "status": "ok" if support.eligible_for_discrimination else "insufficient",
        "support_status": support.status,
        "support_eligible_for_discrimination": support.eligible_for_discrimination,
        "insufficient_reason": support.reason,
        "n_test": int(len(y_test)),
        "test_positive_n": positive_n,
        "test_negative_n": negative_n,
        "test_positive_rate": float(y_test.mean()) if len(y_test) else float("nan"),
        "patient_cluster_n": int(pd.Series(groups).nunique()),
        "bootstrap_group_source": group_source,
        "best_hyperparameters": _canonical_parameter_json(fitted.best_parameters),
        "development_membership_sha256": fitted.development_membership_sha256,
        "fold_membership_sha256": fitted.fold_membership_sha256,
        "evaluation_membership_sha256": membership_sha256(frame),
        "valid_fold_n": int(fitted.tuning["fold_index"].nunique()),
        "oof_n": int(len(fitted.oof_predictions)),
        "final_fit_converged": fitted.final_fit_converged,
        "final_fit_iterations": fitted.final_fit_iterations,
        "standardization": "StandardScaler(with_mean=False)",
        "calibration": "none_raw_probability_only",
        "target_labels_used_for_training": False,
        "target_labels_used_for_hyperparameter_tuning": False,
        "target_labels_used_for_calibration": False,
        **metrics,
    }
    if extra_fields:
        row.update(dict(extra_fields))

    bootstrap = pd.DataFrame()
    if support.eligible_for_discrimination and bootstrap_count > 0:
        bootstrap = _cluster_bootstrap_intervals(
            y_test,
            probability,
            None,
            groups,
            {},
            n_boot=bootstrap_count,
            seed=seed,
        )
        if not bootstrap.empty:
            bootstrap.insert(0, "site_id", site_id)
            bootstrap.insert(0, "cohort_id", cohort_id)
            bootstrap.insert(0, "task_id", fitted.task_id)
            bootstrap.insert(0, "protocol_family", protocol_family)
            for interval in bootstrap.itertuples(index=False):
                row[f"{interval.metric}_ci_low"] = interval.ci_low
                row[f"{interval.metric}_ci_high"] = interval.ci_high

    prediction_columns = ["public_sample_id", "y"]
    if "public_patient_cluster_id" in frame:
        prediction_columns.insert(1, "public_patient_cluster_id")
    predictions = frame[prediction_columns].copy()
    predictions.insert(0, "site_id", site_id)
    predictions.insert(0, "cohort_id", cohort_id)
    predictions.insert(0, "task_id", fitted.task_id)
    predictions.insert(0, "protocol_family", protocol_family)
    predictions["raw_probability"] = probability
    return row, bootstrap, predictions


def cohort_audit_row(
    frame: pd.DataFrame,
    *,
    protocol_family: str,
    task_id: str,
    cohort_id: str,
    site_id: str,
    role: str,
) -> dict[str, object]:
    """Create a de-identified cohort-membership audit row."""

    y = frame.get("y", pd.Series(dtype=np.int8)).to_numpy(dtype=np.int8)
    return {
        "protocol_family": protocol_family,
        "task_id": task_id,
        "cohort_id": cohort_id,
        "site_id": site_id,
        "role": role,
        "n": int(len(frame)),
        "positive_n": int(y.sum()) if len(y) else 0,
        "negative_n": int(y.size - y.sum()) if len(y) else 0,
        "membership_sha256": membership_sha256(frame),
    }
