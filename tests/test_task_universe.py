from __future__ import annotations

import pandas as pd
import pytest

from zd_mast.task_universe import (
    SupportThresholds,
    feature_available_samples,
    normalize_exact_identity,
    run_task_universe_audit,
    validate_core_task_config,
)


def core_task_config() -> list[dict[str, str]]:
    literature = ["sa_oxa", "kp_cro", "ec_cro"]
    local = ["sa_lvx", "sa_gen", "kp_fep", "kp_caz", "kp_cip", "ec_cip", "ec_fep"]
    tasks: list[dict[str, str]] = []
    for task_id in literature:
        tasks.append(
            {
                "task_id": task_id,
                "organism": task_id.split("_")[0],
                "antimicrobial": task_id.split("_")[1],
                "historical_provenance_category": "literature_anchor",
                "original_human_selection_evidence": "",
            }
        )
    for task_id in local:
        tasks.append(
            {
                "task_id": task_id,
                "organism": task_id.split("_")[0],
                "antimicrobial": task_id.split("_")[1],
                "historical_provenance_category": "local_extension",
                "original_human_selection_evidence": "",
            }
        )
    return tasks


def synthetic_ast() -> pd.DataFrame:
    rows = [
        # This exact combination passes n, class and year support after feature linkage.
        ("A", "S1", 2023, "Escherichia coli", "FEP", "cefepime", "S", False, "ec_fep"),
        ("A", "S2", 2024, "Escherichia coli", "FEP", "cefepime", "R", False, "ec_fep"),
        # A labelled sample without an available feature row is excluded before thresholds.
        ("A", "S3", 2024, "Escherichia coli", "FEP", "cefepime", "R", False, "ec_fep"),
        # Two categories on one sample-combination are collapsed and flagged as conflict.
        ("A", "S4", 2024, "Escherichia coli", "FEP", "cefepime", "S", False, "ec_fep"),
        ("A", "S4", 2024, "Escherichia coli", "FEP", "cefepime", "R", False, "ec_fep"),
        # Similar antimicrobial names remain separate exact identities.
        ("A", "S5", 2024, "Klebsiella pneumoniae", "CAZ", "ceftazidime", "S", False, "kp_caz"),
        (
            "A",
            "S6",
            2024,
            "Klebsiella pneumoniae",
            "CZA",
            "ceftazidime-avibactam",
            "R",
            False,
            "",
        ),
        # Missing organism identity remains in the first funnel step only.
        ("A", "S7", 2024, "", "FEP", "cefepime", "S", False, ""),
        # Out-of-scope site row.
        ("B", "B1", 2024, "Escherichia coli", "FEP", "cefepime", "S", False, "ec_fep"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "site_id",
            "public_sample_id",
            "year",
            "organism_reported",
            "antimicrobial_code_reported",
            "antimicrobial_name_reported",
            "reported_sir",
            "conflict_flag",
            "task_id_if_core",
        ],
    )


def synthetic_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "public_sample_id": ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "B1"],
            "feature_row": [0, 1, -1, 2, 3, 4, 5, 6],
        }
    )


def test_exact_identity_normalization_never_applies_substring_mapping() -> None:
    assert normalize_exact_identity("  Cefepime  ") == "cefepime"
    assert normalize_exact_identity("Ｃｅｆｅｐｉｍｅ") == "cefepime"
    assert normalize_exact_identity("ceftazidime") != normalize_exact_identity(
        "ceftazidime-avibactam"
    )


def test_feature_availability_accepts_zero_index_and_rejects_negative_sentinel() -> None:
    available = feature_available_samples(synthetic_metadata())
    assert "S1" in available
    assert "S3" not in available


def test_support_funnel_handles_conflicts_linkage_years_and_classes() -> None:
    audit = run_task_universe_audit(
        synthetic_ast(),
        synthetic_metadata(),
        thresholds=SupportThresholds(min_total=2, min_class=1, min_years=2),
        core_tasks=core_task_config(),
        default_author_input_reason="author record required",
        sites=["A"],
        year_min=2023,
        year_max=2024,
    )
    inventory = audit.combination_inventory
    cefepime = inventory.loc[
        inventory["antimicrobial_name_exact_key"].eq("cefepime")
        & inventory["organism_exact_key"].eq("escherichia coli")
    ].iloc[0]
    assert cefepime["raw_sample_combination_n"] == 4
    assert cefepime["conflict_n"] == 1
    assert cefepime["feature_linked_label_n"] == 2
    assert cefepime["total_n"] == 2
    assert cefepime["positive_n_s_vs_ir"] == 1
    assert cefepime["negative_n_s_vs_ir"] == 1
    assert cefepime["year_n"] == 2
    assert bool(cefepime["support_eligible"])

    assert len(inventory) == 3
    assert (
        inventory["antimicrobial_name_exact_key"].isin(
            ["ceftazidime", "ceftazidime-avibactam"]
        ).sum()
        == 2
    )
    funnel = audit.funnel.set_index("step")
    assert funnel.loc["scoped_linked_ast_rows", "label_rows"] == 8
    assert funnel.loc["complete_exact_identity", "label_rows"] == 7
    assert funnel.loc["unique_sample_combination", "label_rows"] == 6
    assert funnel.loc["conflict_free", "label_rows"] == 5
    assert funnel.loc["feature_linked", "label_rows"] == 4
    assert funnel.loc["minimum_class_support", "combination_n"] == 1


def test_core_mapping_records_three_plus_seven_provenance_and_author_gap() -> None:
    audit = run_task_universe_audit(
        synthetic_ast(),
        synthetic_metadata(),
        thresholds=SupportThresholds(min_total=2, min_class=1, min_years=2),
        core_tasks=core_task_config(),
        default_author_input_reason="author record required",
        sites=["A"],
    )
    mapping = audit.core_ten_mapping
    assert mapping["historical_provenance_category"].value_counts().to_dict() == {
        "local_extension": 7,
        "literature_anchor": 3,
    }
    assert mapping["author_input_needed"].all()
    ec_fep = mapping.loc[mapping["task_id"].eq("ec_fep")].iloc[0]
    assert bool(ec_fep["observed_in_task_universe"])
    assert ec_fep["support_eligible_exact_combination_n"] == 1
    assert "not_prospectively_selected" in ec_fep["selection_status"]


def test_invalid_historical_provenance_contract_is_rejected() -> None:
    invalid = core_task_config()
    invalid[0]["historical_provenance_category"] = "local_extension"
    with pytest.raises(ValueError, match="3 literature_anchor"):
        validate_core_task_config(invalid)
