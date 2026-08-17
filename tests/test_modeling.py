from __future__ import annotations

import json

import pandas as pd

from zd_mast.modeling import (
    compare_sensitivities_to_frozen,
    lightgbm_grid,
    primary_run_id,
    stable_seed,
)


def test_frozen_lightgbm_grid_has_twelve_candidates() -> None:
    grid = lightgbm_grid()
    assert len(grid) == 12
    assert {row["num_leaves"] for row in grid} == {15, 31, 63}
    assert {row["min_child_samples"] for row in grid} == {20, 50}
    assert {row["class_weight"] for row in grid} == {None, "balanced"}


def test_primary_run_id_preserves_legacy_seed_contract() -> None:
    run_id = primary_run_id("sa_gen")
    assert run_id == (
        "sa_gentamicin__historical_S_vs_IR__lightgbm__"
        "B_post_marker_current_temporal__all_samples_workload_primary"
    )
    assert stable_seed(run_id) == stable_seed(run_id)


def test_grid_is_json_serializable_with_stable_key_order() -> None:
    assert json.loads(json.dumps(lightgbm_grid()[0], sort_keys=True))["num_leaves"] == 15


def _sensitivity_row(task_id: str, protocol: str) -> dict[str, object]:
    row: dict[str, object] = {
        "task_id": task_id,
        "model": "lightgbm",
        "protocol": protocol,
        "analysis_variant": "patient_disjoint_test",
    }
    for field in (
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
    ):
        row[field] = 10
    for field in (
        "AUROC",
        "AUPRC",
        "AUPRC_baseline",
        "AUPRC_lift",
        "Brier_platt",
        "ECE_platt",
        "sensitivity_sens90",
        "specificity_sens90",
        "non_susceptible_miss_rate_sens90",
    ):
        row[field] = 0.5
    return row


def test_sensitivity_regression_accepts_public_frozen_schema(tmp_path) -> None:
    reproduced = pd.DataFrame([_sensitivity_row("sa_oxa", "current_workflow_protocol_b")])
    frozen = reproduced.copy()
    frozen["endpoint"] = "historical_S_vs_IR"
    path = tmp_path / "public.csv"
    frozen.to_csv(path, index=False)

    regression = compare_sensitivities_to_frozen(reproduced, path)

    assert regression["status"].eq("PASS").all()


def test_sensitivity_regression_accepts_legacy_frozen_schema(tmp_path) -> None:
    reproduced = pd.DataFrame([_sensitivity_row("sa_oxa", "current_workflow_protocol_b")])
    frozen = pd.DataFrame(
        [_sensitivity_row("sa_oxacillin", "B_post_marker_current_temporal")]
    )
    frozen["label_type"] = "historical_S_vs_IR"
    path = tmp_path / "legacy.csv"
    frozen.to_csv(path, index=False)

    regression = compare_sensitivities_to_frozen(reproduced, path)

    assert regression["status"].eq("PASS").all()
