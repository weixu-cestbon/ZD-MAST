import pandas as pd

from zd_mast.harmonization import compare_m100_agreement, summarize_m100_agreement


def test_summarize_m100_agreement_keeps_unsupported_rows_in_denominator() -> None:
    labels = pd.DataFrame(
        {
            "task_id": ["sa_oxa"] * 3,
            "historical_sir": ["S", "R", "S"],
            "historical_binary_s_vs_ir": [0, 1, 0],
            "harmonized_sir": ["R", "R", ""],
            "harmonized_binary_s_vs_ir": [1.0, 1.0, None],
            "reclassified_flag": [True, False, False],
        }
    )

    result = summarize_m100_agreement(labels).iloc[0]

    assert result["total_labels"] == 3
    assert result["interpreted_n"] == 2
    assert result["unsupported_n"] == 1
    assert result["binary_s_vs_ir_agreement"] == 0.5
    assert result["exact_sir_agreement"] == 0.5
    assert result["reclassified_n"] == 1


def test_compare_m100_agreement_reports_changed_value() -> None:
    reproduced = pd.DataFrame(
        {"task_id": ["ec_fep"], "total_labels": [10], "exact_sir_agreement": [1.0]}
    )
    frozen = reproduced.copy()
    frozen.loc[0, "exact_sir_agreement"] = 0.9

    result = compare_m100_agreement(reproduced, frozen)

    assert result.loc[result["field"].eq("total_labels"), "status"].item() == "PASS"
    assert result.loc[result["field"].eq("exact_sir_agreement"), "status"].item() == "FAIL"
