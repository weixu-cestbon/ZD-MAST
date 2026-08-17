from __future__ import annotations

import numpy as np
import pandas as pd

from zd_mast.modeling import PublicProtocolData
from zd_mast.patient_first import build_patient_first_protocol, first_record_per_patient


def frame(rows: list[tuple[str, str | None, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["public_sample_id", "public_patient_cluster_id", "row_order", "y"],
    ).assign(feature_row=lambda x: np.arange(len(x)))


def test_first_record_per_patient_excludes_missing_and_keeps_earliest() -> None:
    data = frame(
        [
            ("s2", "p1", 2, 1),
            ("s1", "p1", 1, 0),
            ("s3", "p2", 3, 1),
            ("s4", None, 4, 0),
        ]
    )
    selected = first_record_per_patient(data)
    assert selected["public_sample_id"].tolist() == ["s1", "s3"]


def test_protocol_purges_seen_test_patients_and_deduplicates_test() -> None:
    development = frame(
        [("d1", "p1", 1, 0), ("d2", "p1", 2, 1), ("d3", "p2", 3, 1)]
    )
    test = frame(
        [("t1", "p1", 4, 0), ("t2", "p3", 5, 1), ("t3", "p3", 6, 0)]
    )
    fold_train = frame(
        [(f"a{i}", f"pa{i}", i, i % 2) for i in range(1, 61)]
    )
    fold_validation = frame(
        [(f"b{i}", f"pb{i}", i + 100, i % 2) for i in range(1, 31)]
    )
    source = PublicProtocolData(
        task_id="sa_oxa",
        feature_matrix=np.zeros((100, 6000), dtype=np.float32),
        development=development,
        test=test,
        folds=[(fold_train, fold_validation, "synthetic")],
    )
    patient_first, audit, _ = build_patient_first_protocol(source)
    assert patient_first.development["public_sample_id"].tolist() == ["d1", "d3"]
    assert patient_first.test["public_sample_id"].tolist() == ["t2"]
    assert audit.test_removed_seen_patient_n == 1
