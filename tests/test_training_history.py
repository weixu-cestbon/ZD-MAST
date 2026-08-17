from __future__ import annotations

import numpy as np
import pandas as pd

from zd_mast.training_history import paired_history_deltas


def test_paired_history_delta_uses_aligned_common_test() -> None:
    rows = []
    for regime, probabilities in {
        "current_workflow_only": [0.1, 0.2, 0.8, 0.9],
        "pre_marker_history_only": [0.2, 0.3, 0.7, 0.8],
        "pooled_pre_and_current": [0.05, 0.1, 0.85, 0.95],
    }.items():
        for index, probability in enumerate(probabilities):
            rows.append(
                {
                    "task_id": "sa_oxa",
                    "training_regime": regime,
                    "analysis_variant": "patient_disjoint_common_test_primary",
                    "public_sample_id": f"S{index}",
                    "public_patient_cluster_id": f"P{index}",
                    "y": int(index >= 2),
                    "raw_probability": probability,
                }
            )
    output = paired_history_deltas(pd.DataFrame(rows), bootstrap_count=20, seed=7)
    assert set(output["comparator_regime"]) == {
        "pre_marker_history_only",
        "pooled_pre_and_current",
    }
    assert np.isfinite(output["delta_comparator_minus_current"]).all()


def test_paired_history_delta_rejects_misaligned_rows() -> None:
    rows = []
    for regime in (
        "current_workflow_only",
        "pre_marker_history_only",
        "pooled_pre_and_current",
    ):
        for index in range(4):
            if regime == "pre_marker_history_only" and index == 3:
                continue
            rows.append(
                {
                    "task_id": "sa_oxa",
                    "training_regime": regime,
                    "analysis_variant": "patient_disjoint_common_test_primary",
                    "public_sample_id": f"S{index}",
                    "public_patient_cluster_id": f"P{index}",
                    "y": int(index >= 2),
                    "raw_probability": (index + 1) / 5,
                }
            )
    try:
        paired_history_deltas(pd.DataFrame(rows), bootstrap_count=10, seed=7)
    except ValueError as error:
        assert "not aligned" in str(error)
    else:
        raise AssertionError("Expected misaligned prediction rows to fail")
