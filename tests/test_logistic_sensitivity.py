from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from zd_mast.logistic_sensitivity import (
    CLASS_WEIGHT_GRID,
    C_GRID,
    evaluate_logistic_fit,
    fit_development_logistic,
    fit_logistic_model,
    fold_membership_sha256,
    logistic_grid,
    membership_sha256,
    tune_logistic_with_frozen_folds,
    validate_development_contract,
)


def _synthetic_contract() -> tuple[
    np.ndarray,
    pd.DataFrame,
    list[tuple[pd.DataFrame, pd.DataFrame, str]],
    pd.DataFrame,
]:
    rng = np.random.default_rng(19)
    n_rows = 120
    n_features = 12
    matrix = rng.normal(size=(n_rows, n_features)).astype(np.float32)
    signal = matrix[:, 0] - 0.7 * matrix[:, 1] + 0.3 * matrix[:, 2]
    y = (signal > np.median(signal)).astype(np.int8)
    frame = pd.DataFrame(
        {
            "public_sample_id": [f"S{index:03d}" for index in range(n_rows)],
            "public_patient_cluster_id": [f"P{index:03d}" for index in range(n_rows)],
            "feature_row": np.arange(n_rows, dtype=int),
            "y": y,
        }
    )
    development = frame.iloc[:80].copy().reset_index(drop=True)
    fold_one_train = development.iloc[:40].copy().reset_index(drop=True)
    fold_one_validation = development.iloc[40:60].copy().reset_index(drop=True)
    fold_two_train = development.iloc[:60].copy().reset_index(drop=True)
    fold_two_validation = development.iloc[60:80].copy().reset_index(drop=True)
    # Guarantee both classes in every synthetic fold without changing features.
    for split in (
        fold_one_train,
        fold_one_validation,
        fold_two_train,
        fold_two_validation,
    ):
        if split["y"].nunique() < 2:
            split.loc[split.index[0], "y"] = 0
            split.loc[split.index[-1], "y"] = 1
    development = pd.concat(
        [fold_one_train, fold_one_validation, fold_two_validation],
        ignore_index=True,
    ).drop_duplicates("public_sample_id", keep="last")
    development = development.sort_values("feature_row", kind="stable").reset_index(drop=True)
    folds = [
        (fold_one_train, fold_one_validation, "fold=1"),
        (fold_two_train, fold_two_validation, "fold=2"),
    ]
    evaluation = frame.iloc[80:].copy().reset_index(drop=True)
    return matrix, development, folds, evaluation


def test_logistic_grid_is_exact_prespecified_eight_candidates() -> None:
    grid = logistic_grid()
    assert len(grid) == 8
    assert {row["C"] for row in grid} == set(C_GRID)
    assert {row["class_weight"] for row in grid} == set(CLASS_WEIGHT_GRID)
    assert {row["penalty"] for row in grid} == {"l2"}
    assert {row["solver"] for row in grid} == {"liblinear"}


def test_sparse_standardization_never_centers() -> None:
    x = sparse.csr_matrix(
        np.asarray(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 2.0, 1.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    y = np.asarray([0, 1, 0, 1], dtype=np.int8)
    model, audit = fit_logistic_model(x, y, logistic_grid()[0], seed=3)
    assert model.named_steps["scale"].with_mean is False
    assert audit["sparse_centering_disabled"] is True
    probability = model.predict_proba(x)[:, 1]
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()


def test_tuning_uses_all_frozen_folds_and_selects_one_candidate() -> None:
    matrix, development, folds, evaluation = _synthetic_contract()
    validate_development_contract(development, folds, evaluation_frames=(evaluation,))
    best, tuning = tune_logistic_with_frozen_folds(matrix, folds, seed=11)
    assert len(tuning) == 16
    assert tuning["fold_index"].nunique() == 2
    assert tuning.loc[tuning["selected_candidate"], "candidate_index"].nunique() == 1
    assert best["C"] in C_GRID
    assert best["class_weight"] in CLASS_WEIGHT_GRID
    assert np.isfinite(tuning[["AUROC", "AUPRC"]].to_numpy()).all()


def test_development_contract_rejects_evaluation_leakage() -> None:
    _matrix, development, folds, evaluation = _synthetic_contract()
    leaked = evaluation.copy()
    leaked.loc[0, "public_sample_id"] = development.loc[0, "public_sample_id"]
    with pytest.raises(ValueError, match="Development/evaluation sample overlap"):
        validate_development_contract(development, folds, evaluation_frames=(leaked,))


def test_evaluation_does_not_refit_or_use_target_labels_for_probability() -> None:
    matrix, development, folds, evaluation = _synthetic_contract()
    fitted = fit_development_logistic(
        matrix,
        "synthetic_task",
        development,
        folds,
        seed=23,
        evaluation_frames_for_overlap_audit=(evaluation,),
    )
    coefficients_before = fitted.model.named_steps["logistic"].coef_.copy()
    row_one, _bootstrap, prediction_one = evaluate_logistic_fit(
        fitted,
        matrix,
        evaluation,
        protocol_family="synthetic_source_only",
        cohort_id="target",
        site_id="ZD-MAST-B",
        feature_representation="peak_presence6000",
        bootstrap_count=0,
        seed=5,
    )
    flipped = evaluation.copy()
    flipped["y"] = 1 - flipped["y"]
    row_two, _bootstrap, prediction_two = evaluate_logistic_fit(
        fitted,
        matrix,
        flipped,
        protocol_family="synthetic_source_only",
        cohort_id="target",
        site_id="ZD-MAST-B",
        feature_representation="peak_presence6000",
        bootstrap_count=0,
        seed=5,
    )
    np.testing.assert_allclose(
        prediction_one["raw_probability"], prediction_two["raw_probability"]
    )
    np.testing.assert_allclose(
        coefficients_before,
        fitted.model.named_steps["logistic"].coef_,
    )
    assert row_one["target_labels_used_for_training"] is False
    assert row_one["target_labels_used_for_hyperparameter_tuning"] is False
    assert row_one["target_labels_used_for_calibration"] is False
    assert row_one["raw_auroc"] != row_two["raw_auroc"]


def test_membership_hashes_are_stable_and_order_sensitive() -> None:
    _matrix, development, folds, _evaluation = _synthetic_contract()
    assert membership_sha256(development) == membership_sha256(development.copy())
    assert membership_sha256(development) != membership_sha256(
        development.iloc[::-1].reset_index(drop=True)
    )
    assert fold_membership_sha256(folds) == fold_membership_sha256(list(folds))
    changed = [(folds[0][0], folds[0][1].iloc[::-1].reset_index(drop=True), folds[0][2]), folds[1]]
    assert fold_membership_sha256(folds) != fold_membership_sha256(changed)
