from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import zd_mast.training_history_sensitivity as sensitivity
from zd_mast.training_history import HistoryTaskCohorts, TrainingHistoryInputs
from zd_mast.training_history_sensitivity import (
    CURRENT_ONLY,
    POOLED_ERA_BALANCED,
    PRE_MARKER_ONLY,
    REQUIRED_LEARNING_FRACTIONS,
    TRAINING_REGIMES,
    build_repeated_size_matched_cells,
    paired_size_matched_deltas,
    run_size_matched_task,
    summarize_paired_deltas,
    validate_fixed_patient_disjoint_test,
)


def synthetic_pool(
    prefix: str,
    *,
    negative_n: int,
    positive_n: int,
    feature_start: int = 0,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    feature_row = feature_start
    for label, count in ((0, negative_n), (1, positive_n)):
        for index in range(count):
            sample_id = f"{prefix}_Y{label}_{index:03d}"
            rows.append(
                {
                    "public_sample_id": sample_id,
                    "public_patient_cluster_id": f"{prefix}_P{label}_{index:03d}",
                    "feature_row": feature_row,
                    "row_order": feature_row,
                    "y": label,
                }
            )
            feature_row += 1
    return pd.DataFrame(rows)


def cells_by_key(
    cells: list[sensitivity.SampledTrainingCell],
) -> dict[tuple[int, float, str], sensitivity.SampledTrainingCell]:
    return {
        (cell.repeat_index, cell.learning_fraction, cell.training_regime): cell
        for cell in cells
    }


def test_repeated_sampler_is_deterministic_nested_class_matched_and_era_balanced() -> None:
    pre = synthetic_pool("PRE", negative_n=40, positive_n=28)
    current = synthetic_pool("CUR", negative_n=24, positive_n=36, feature_start=1000)
    first, first_audit, caps = build_repeated_size_matched_cells(
        pre,
        current,
        task_id="sa_oxa",
        repeats=2,
        seed=41,
        minimum_train_class_n=1,
    )
    second, second_audit, _ = build_repeated_size_matched_cells(
        pre,
        current,
        task_id="sa_oxa",
        repeats=2,
        seed=41,
        minimum_train_class_n=1,
    )

    assert first_audit["training_sample_signature"].tolist() == second_audit[
        "training_sample_signature"
    ].tolist()
    assert caps.set_index("class_label")["common_class_cap_n"].to_dict() == {
        0: 24,
        1: 28,
    }

    indexed = cells_by_key(first)
    for repeat_index in range(2):
        previous_ids = {regime: set() for regime in TRAINING_REGIMES}
        for fraction in REQUIRED_LEARNING_FRACTIONS:
            cells_at_fraction = [
                indexed[(repeat_index, fraction, regime)] for regime in TRAINING_REGIMES
            ]
            class_counts = [
                tuple(
                    int(cell.frame["y"].eq(label).sum())
                    for label in (0, 1)
                )
                for cell in cells_at_fraction
            ]
            assert len(set(class_counts)) == 1
            for cell in cells_at_fraction:
                sample_ids = set(cell.frame["public_sample_id"])
                assert previous_ids[cell.training_regime].issubset(sample_ids)
                previous_ids[cell.training_regime] = sample_ids

            pooled = indexed[(repeat_index, fraction, POOLED_ERA_BALANCED)].frame
            for label in (0, 1):
                era_counts = pooled.loc[pooled["y"].eq(label), "source_era"].value_counts()
                assert abs(
                    int(era_counts.get("pre_marker", 0))
                    - int(era_counts.get("current_workflow", 0))
                ) <= 1


def test_different_repeats_change_source_samples_but_not_target_counts() -> None:
    pre = synthetic_pool("PRE", negative_n=40, positive_n=40)
    current = synthetic_pool("CUR", negative_n=40, positive_n=40, feature_start=1000)
    cells, audit, _ = build_repeated_size_matched_cells(
        pre,
        current,
        task_id="kp_fep",
        repeats=2,
        seed=7,
        minimum_train_class_n=1,
    )
    quarter_current = audit.loc[
        audit["learning_fraction"].eq(0.25)
        & audit["training_regime"].eq(CURRENT_ONLY)
    ].sort_values("repeat_index")
    assert quarter_current["training_sample_signature"].nunique() == 2
    assert quarter_current["n_development"].nunique() == 1
    assert quarter_current["development_positive_n"].nunique() == 1
    assert len(cells) == 2 * 4 * 3


def test_fixed_future_test_rejects_patient_overlap() -> None:
    pre = synthetic_pool("PRE", negative_n=4, positive_n=4)
    current = synthetic_pool("CUR", negative_n=4, positive_n=4, feature_start=100)
    test = synthetic_pool("TEST", negative_n=2, positive_n=2, feature_start=200)
    test.loc[0, "public_patient_cluster_id"] = pre.loc[
        0, "public_patient_cluster_id"
    ]
    with pytest.raises(ValueError, match="patient overlap"):
        validate_fixed_patient_disjoint_test(pre, current, test, task_id="ec_cip")


def synthetic_metric_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    values = {
        0: {
            CURRENT_ONLY: (0.60, 0.40),
            PRE_MARKER_ONLY: (0.70, 0.45),
            POOLED_ERA_BALANCED: (0.75, 0.50),
        },
        1: {
            CURRENT_ONLY: (0.62, 0.42),
            PRE_MARKER_ONLY: (0.68, 0.44),
            POOLED_ERA_BALANCED: (0.76, 0.52),
        },
    }
    for repeat_index, regime_values in values.items():
        for regime, (auroc, auprc) in regime_values.items():
            rows.append(
                {
                    "task_id": "sa_oxa",
                    "training_regime": regime,
                    "repeat_index": repeat_index,
                    "learning_fraction": 1.0,
                    "status": "ok",
                    "n_development": 80,
                    "development_positive_n": 40,
                    "development_negative_n": 40,
                    "n_test": 20,
                    "fixed_test_signature": "fixed-test",
                    "raw_auroc": auroc,
                    "raw_auprc": auprc,
                }
            )
    return pd.DataFrame(rows)


def test_paired_deltas_include_pooled_vs_pre_and_repeat_summary() -> None:
    deltas = paired_size_matched_deltas(synthetic_metric_rows())
    assert "pooled_era_balanced_minus_pre_marker_only" in set(
        deltas["comparison_id"]
    )
    pooled_pre_auroc = deltas.loc[
        deltas["comparison_id"].eq(
            "pooled_era_balanced_minus_pre_marker_only"
        )
        & deltas["metric"].eq("raw_auroc")
    ].sort_values("repeat_index")
    np.testing.assert_allclose(
        pooled_pre_auroc["delta_comparator_minus_reference"], [0.05, 0.08]
    )
    summary = summarize_paired_deltas(deltas)
    summary_row = summary.loc[
        summary["comparison_id"].eq(
            "pooled_era_balanced_minus_pre_marker_only"
        )
        & summary["metric"].eq("raw_auroc")
    ].iloc[0]
    assert int(summary_row["repeat_n"]) == 2
    assert np.isclose(summary_row["delta_median"], 0.065)


def test_insufficient_cells_produce_machine_readable_empty_delta_tables() -> None:
    metrics = synthetic_metric_rows()
    metrics["status"] = "insufficient"
    metrics["raw_auroc"] = np.nan
    metrics["raw_auprc"] = np.nan
    deltas = paired_size_matched_deltas(metrics)
    summary = summarize_paired_deltas(deltas)
    assert deltas.empty
    assert summary.empty
    assert "delta_comparator_minus_reference" in deltas.columns
    assert "repeat_distribution_q025" in summary.columns


def test_task_runner_uses_frozen_parameters_and_identical_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pre = synthetic_pool("PRE", negative_n=8, positive_n=8)
    current = synthetic_pool("CUR", negative_n=8, positive_n=8, feature_start=16)
    pre.loc[0, "public_patient_cluster_id"] = pd.NA
    current.loc[0, "public_patient_cluster_id"] = ""
    test = synthetic_pool("TEST", negative_n=2, positive_n=2, feature_start=32)
    matrix = np.zeros((36, 6000), dtype=np.float32)
    matrix[test.loc[test["y"].eq(0), "feature_row"].to_numpy(), 0] = 0.1
    matrix[test.loc[test["y"].eq(1), "feature_row"].to_numpy(), 0] = 0.9
    frozen_parameters = {
        "num_leaves": 15,
        "learning_rate": 0.05,
        "n_estimators": 300,
    }
    inputs = TrainingHistoryInputs(
        feature_root=Path("synthetic"),
        matrix=matrix,
        bridge=pd.DataFrame(),
        current_folds=pd.DataFrame(),
        fixed_parameters={"sa_oxa": frozen_parameters},
    )
    cohorts = HistoryTaskCohorts(
        task_id="sa_oxa",
        development_by_regime={
            "pre_marker_history_only": pre,
            "current_workflow_only": current,
            "pooled_pre_and_current": pd.concat([pre, current], ignore_index=True),
        },
        folds_by_regime={},
        fold_audit=pd.DataFrame(),
        test_all_samples=test,
        test_patient_disjoint_common=test,
        test_purge_audit={},
    )
    fitted: list[tuple[dict[str, object], int]] = []

    def fake_fit(
        params: dict[str, object],
        _x: object,
        _y: np.ndarray,
        seed: int,
        _threads: int,
    ) -> dict[str, int]:
        fitted.append((dict(params), seed))
        return {"seed": seed}

    def fake_predict(_model: object, x: object) -> np.ndarray:
        return np.asarray(x[:, 0].toarray()).ravel()

    monkeypatch.setattr(sensitivity, "fit_lightgbm", fake_fit)
    monkeypatch.setattr(sensitivity, "predict", fake_predict)
    metrics, audit, caps, predictions = run_size_matched_task(
        inputs,
        cohorts,
        repeats=1,
        seed=13,
        minimum_train_class_n=1,
        return_predictions=True,
    )

    assert len(metrics) == 4 * 3
    assert metrics["status"].eq("ok").all()
    assert metrics["fixed_test_signature"].nunique() == 1
    assert metrics["raw_auroc"].eq(1.0).all()
    assert metrics["raw_auprc"].eq(1.0).all()
    assert not metrics["test_labels_used_for_sampling_or_tuning"].any()
    assert not metrics["calibration_applied"].any()
    assert not metrics["threshold_selection_applied"].any()
    assert metrics["pre_missing_patient_excluded_n"].eq(1).all()
    assert metrics["current_missing_patient_excluded_n"].eq(1).all()
    assert audit["pre_missing_patient_excluded_n"].eq(1).all()
    assert audit["current_missing_patient_excluded_n"].eq(1).all()
    assert caps["pre_missing_patient_excluded_n"].eq(1).all()
    assert caps["current_missing_patient_excluded_n"].eq(1).all()
    assert all(params == frozen_parameters for params, _ in fitted)
    assert len(audit) == len(metrics)
    assert set(caps["class_label"]) == {0, 1}
    assert not predictions.empty
