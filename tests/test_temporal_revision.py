from __future__ import annotations

import pandas as pd

from zd_mast.temporal_revision import build_year_folds, purge_by_training_patients, support_status


def row(sample: str, patient: str | None, year: int, y: int, order: int) -> dict[str, object]:
    return {
        "public_sample_id": sample,
        "public_patient_cluster_id": patient,
        "year": year,
        "y": y,
        "row_order": order,
    }


def test_patient_purge_removes_overlap_and_missing() -> None:
    train = pd.DataFrame([row("A", "P1", 2019, 0, 0), row("B", "P2", 2019, 1, 1)])
    test = pd.DataFrame(
        [
            row("C", "P1", 2020, 0, 0),
            row("D", None, 2020, 1, 1),
            row("E", "P3", 2020, 1, 2),
        ]
    )
    purged, audit = purge_by_training_patients(train, test)
    assert purged["public_sample_id"].tolist() == ["E"]
    assert audit["removed_patient_overlap_n"] == 1
    assert audit["removed_missing_patient_cluster_n"] == 1


def test_support_status_distinguishes_absent_and_insufficient() -> None:
    empty = pd.DataFrame(columns=["y"])
    assert support_status(empty, 100, 20)[0] == "absent"
    small = pd.DataFrame({"y": [0, 1]})
    assert support_status(small, 100, 20)[0] == "insufficient"


def test_year_folds_are_temporal_and_patient_purged() -> None:
    rows = []
    order = 0
    for year in (2019, 2020, 2021):
        for index in range(20):
            patient = f"P{year}_{index}"
            rows.append(row(f"S{year}_{index}", patient, year, index % 2, order))
            order += 1
    rows.append(row("overlap", "P2019_0", 2020, 1, order))
    development = pd.DataFrame(rows)
    folds, audit = build_year_folds(
        development,
        minimum_train_total=10,
        minimum_train_class=2,
        minimum_validation_total=10,
        minimum_validation_class=2,
    )
    assert len(folds) == 2
    assert set(folds[0][0]["year"]) == {2019}
    assert set(folds[0][1]["year"]) == {2020}
    assert "overlap" not in set(folds[0][1]["public_sample_id"])
    assert int(audit.loc[audit["validation_year"].eq(2020), "removed_patient_overlap_n"].iloc[0]) == 1
