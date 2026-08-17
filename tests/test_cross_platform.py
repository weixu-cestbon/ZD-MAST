from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zd_mast.cross_platform import (
    _attach_groups,
    attach_target_dates,
    apply_fixed_thresholds,
    classify_support,
    filter_date_window,
    fit_guarded_platt,
    independent_cluster_bootstrap_delta,
    normalize_target_date_table,
    derive_patient_disjoint_test,
    purge_validation_for_patient_disjoint,
    probability_metrics,
    reject_patient_cluster_overlap,
    reject_duplicate_labels,
    reject_source_overlap,
    select_fixed_thresholds,
    site_difference_intervals,
)


def test_target_date_filter_is_inclusive_and_accepts_collection_date() -> None:
    raw = pd.DataFrame(
        {
            "public_sample_id": ["B1", "B2", "B3", "B4"],
            "collection_date": ["2026-02-28", "2026-03-01", "2026-06-09", "2026-06-10"],
        }
    )
    dates, source = normalize_target_date_table(raw)
    selected = filter_date_window(dates, "2026-03-01", "2026-06-09")
    assert source == "collection_date"
    assert selected["public_sample_id"].tolist() == ["B2", "B3"]


def test_target_date_table_accepts_accept_datetime() -> None:
    raw = pd.DataFrame(
        {
            "public_sample_id": ["B1", "B2"],
            "accept_datetime": ["2026-03-01 08:30:00", "2026-03-02 09:00:00"],
        }
    )
    dates, source = normalize_target_date_table(raw)
    assert source == "accept_datetime"
    assert dates["target_date"].dt.hour.eq(0).all()


def test_target_date_table_preserves_randomized_patient_cluster() -> None:
    raw = pd.DataFrame(
        {
            "public_sample_id": ["B1", "B2"],
            "accept_datetime": ["2026-03-01", "2026-03-02"],
            "public_patient_cluster_id": ["ZDMB_PAT_00000001", ""],
        }
    )
    dates, _ = normalize_target_date_table(raw)
    assert dates.columns.tolist() == [
        "public_sample_id",
        "target_date",
        "public_patient_cluster_id",
    ]
    assert dates.loc[0, "public_patient_cluster_id"] == "ZDMB_PAT_00000001"


def test_global_target_groups_coexist_with_task_specific_source_groups() -> None:
    frame = pd.DataFrame({"public_sample_id": ["B1"], "y": [1]})
    groups = pd.DataFrame(
        {
            "public_sample_id": ["A1", "B1"],
            "task_id": pd.Series(["ec_fep", pd.NA], dtype="string"),
            "public_patient_cluster_id": ["ZDMA_PAT_00000001", "ZDMB_PAT_00000001"],
        }
    )
    output = _attach_groups(frame, groups, "ec_fep")
    assert output.loc[0, "public_patient_cluster_id"] == "ZDMB_PAT_00000001"


def test_attaching_target_dates_does_not_suffix_patient_cluster_column() -> None:
    frame = pd.DataFrame(
        {
            "public_sample_id": ["B1"],
            "public_patient_cluster_id": ["ZDMB_PAT_00000001"],
        }
    )
    dates = pd.DataFrame(
        {
            "public_sample_id": ["B1"],
            "target_date": pd.to_datetime(["2026-03-01"]),
            "public_patient_cluster_id": ["ZDMB_PAT_00000001"],
        }
    )
    output = attach_target_dates(frame, dates)
    assert "public_patient_cluster_id" in output
    assert not any(column.endswith(("_x", "_y")) for column in output)


def test_duplicate_labels_and_source_overlap_are_rejected() -> None:
    labels = pd.DataFrame(
        {
            "site_id": ["ZD-MAST-A", "ZD-MAST-A"],
            "public_sample_id": ["A1", "A1"],
            "task_id": ["sa_oxa", "sa_oxa"],
        }
    )
    with pytest.raises(ValueError, match="Duplicate site/sample/task"):
        reject_duplicate_labels(labels)

    development = pd.DataFrame({"public_sample_id": ["A1", "A2"]})
    test = pd.DataFrame({"public_sample_id": ["A2", "A3"]})
    with pytest.raises(ValueError, match="development/test overlap"):
        reject_source_overlap(development, test)


def test_patient_cluster_overlap_is_rejected_when_groups_are_available() -> None:
    development = pd.DataFrame({"public_patient_cluster_id": ["P1", "P2"]})
    test = pd.DataFrame({"public_patient_cluster_id": ["P2", "P3"]})
    with pytest.raises(ValueError, match="patient-cluster overlap"):
        reject_patient_cluster_overlap(development, test)


def test_fold_validation_purge_keeps_training_and_records_reasons() -> None:
    train = pd.DataFrame(
        {
            "public_sample_id": ["A1", "A2", "A3", "A4"],
            "public_patient_cluster_id": ["P1", "P2", "P3", "P4"],
            "y": [0, 1, 0, 1],
        }
    )
    validation = pd.DataFrame(
        {
            "public_sample_id": ["V1", "V2", "V3", "V4"],
            "public_patient_cluster_id": pd.Series(["P2", "P5", None, "P6"], dtype="string"),
            "y": [0, 1, 0, 0],
        }
    )
    train_before = train.copy(deep=True)
    purged, audit = purge_validation_for_patient_disjoint(
        train,
        validation,
        task_id="ec_fep",
        fold_index=0,
        train_minimum_class=1,
        validation_minimum_class=1,
    )
    pd.testing.assert_frame_equal(train, train_before)
    assert purged["public_sample_id"].tolist() == ["V2", "V4"]
    assert audit["removed_overlap_n"] == 1
    assert audit["removed_missing_patient_cluster_n"] == 1
    assert audit["validation_after_n"] == 2
    assert audit["purge_status"] == "PASS"


def test_patient_disjoint_test_purge_is_separate_from_all_sample_test() -> None:
    development = pd.DataFrame(
        {
            "public_sample_id": ["A1", "A2"],
            "public_patient_cluster_id": ["P1", "P2"],
            "y": [0, 1],
        }
    )
    test = pd.DataFrame(
        {
            "public_sample_id": ["T1", "T2", "T3", "T4"],
            "public_patient_cluster_id": pd.Series(["P2", "P3", "P4", None], dtype="string"),
            "y": [1, 1, 0, 0],
        }
    )
    strict, audit = derive_patient_disjoint_test(
        development,
        test,
        task_id="ec_fep",
        minimum_total=2,
        minimum_class=1,
    )
    assert len(test) == 4
    assert strict["public_sample_id"].tolist() == ["T2", "T3"]
    assert audit["removed_overlap_n"] == 1
    assert audit["removed_missing_patient_cluster_n"] == 1
    assert audit["n_before"] == 4
    assert audit["n_after"] == 2
    assert audit["support_status"] == "adequate"


def test_probability_metrics_keeps_explicit_nan_fields_for_single_class() -> None:
    result = probability_metrics(
        np.array([0, 0], dtype=np.int8),
        np.array([0.1, 0.2]),
        np.array([0.15, 0.25]),
        {"youden": 0.5},
    )
    assert result["threshold_metrics_valid"] is False
    assert np.isnan(result["sensitivity_youden"])
    assert np.isnan(result["specificity_specificity90"])


def test_threshold_selection_returns_finite_validation_thresholds() -> None:
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    probability = np.array([0.05, 0.20, 0.40, 0.60, 0.80, 0.95])
    thresholds = select_fixed_thresholds(y, probability)
    assert set(thresholds) == {"youden", "sensitivity90", "specificity90"}
    assert all(np.isfinite(value) for value in thresholds.values())


def test_negative_platt_slope_is_guarded_without_reversing_predictions() -> None:
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    inverted_probability = np.array([0.95, 0.85, 0.75, 0.25, 0.15, 0.05])
    calibration = fit_guarded_platt(y, inverted_probability)
    assert calibration.status == "failed"
    assert calibration.reason == "nonpositive_platt_slope"
    assert calibration.slope < 0
    assert calibration.apply(inverted_probability) is None


def test_fixed_threshold_application_uses_supplied_values() -> None:
    y = np.array([0, 0, 1, 1], dtype=np.int8)
    probability = np.array([0.1, 0.7, 0.6, 0.9])
    metrics = apply_fixed_thresholds(y, probability, {"youden": 0.65})
    assert metrics["threshold_youden"] == pytest.approx(0.65)
    assert metrics["sensitivity_youden"] == pytest.approx(0.5)
    assert metrics["specificity_youden"] == pytest.approx(0.5)
    assert metrics["false_susceptible_rate_youden"] == pytest.approx(0.5)
    assert metrics["false_resistant_rate_youden"] == pytest.approx(0.5)


def test_independent_cluster_bootstrap_shape_and_reproducibility() -> None:
    a_y = np.array([0, 0, 1, 1, 0, 1], dtype=np.int8)
    a_p = np.array([0.1, 0.2, 0.7, 0.9, 0.4, 0.6])
    a_group = np.array(["a1", "a1", "a2", "a3", "a4", "a5"])
    b_y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int8)
    b_p = np.array([0.2, 0.8, 0.3, 0.6, 0.45, 0.55])
    b_group = np.array(["b1", "b2", "b3", "b3", "b4", "b5"])
    first = independent_cluster_bootstrap_delta(
        a_y,
        a_p,
        a_group,
        b_y,
        b_p,
        b_group,
        metric="auroc",
        n_boot=40,
        seed=123,
    )
    second = independent_cluster_bootstrap_delta(
        a_y,
        a_p,
        a_group,
        b_y,
        b_p,
        b_group,
        metric="auroc",
        n_boot=40,
        seed=123,
    )
    assert first.shape == (40, 4)
    pd.testing.assert_frame_equal(first, second)
    assert first["bootstrap_replicate"].tolist() == list(range(40))


def test_primary_site_delta_uses_patient_disjoint_site_a_cohort() -> None:
    source = pd.DataFrame(
        {
            "public_sample_id": ["A1", "A2", "A3", "A4"],
            "public_patient_cluster_id": ["PA1", "PA2", "PA3", "PA4"],
            "y": [0, 1, 0, 1],
            "raw_probability": [0.1, 0.8, 0.3, 0.9],
            "calibrated_probability": [0.1, 0.8, 0.3, 0.9],
        }
    )
    target = pd.DataFrame(
        {
            "public_sample_id": ["B1", "B2", "B3", "B4"],
            "public_patient_cluster_id": ["PB1", "PB2", "PB3", "PB4"],
            "y": [0, 1, 0, 1],
            "raw_probability": [0.2, 0.7, 0.4, 0.85],
            "calibrated_probability": [0.2, 0.7, 0.4, 0.85],
        }
    )
    result = site_difference_intervals(
        "ec_fep",
        source,
        target,
        target_cohort_id="site_b_primary",
        n_boot=12,
        seed=7,
    )
    assert set(result["source_cohort_id"]) == {"site_a_test_patient_disjoint"}
    assert set(result["comparison_role"]) == {"primary"}
    assert set(result["target_cohort_id"]) == {"site_b_primary"}


@pytest.mark.parametrize(
    ("total", "positive", "negative", "status", "eligible"),
    [
        (120, 30, 90, "adequate", True),
        (80, 15, 65, "exploratory_or_insufficient", True),
        (50, 0, 50, "insufficient", False),
        (0, 0, 0, "insufficient", False),
    ],
)
def test_preflight_support_classification(
    total: int,
    positive: int,
    negative: int,
    status: str,
    eligible: bool,
) -> None:
    support = classify_support(total, positive, negative)
    assert support.status == status
    assert support.eligible_for_discrimination is eligible
