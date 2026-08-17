from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from zd_mast.protocols import (
    build_temporal_cohorts,
    episode_first_rows,
    patient_disjoint_test,
    rolling_origin_folds,
)


class ProtocolTests(unittest.TestCase):
    def test_primary_protocol_uses_frozen_calendar_boundaries(self) -> None:
        frame = pd.DataFrame(
            {
                "sample_key": ["a", "b", "c", "d", "e", "f"],
                "date": [
                    "2025-06-22",
                    "2025-06-23",
                    "2025-07-01",
                    "2026-02-28",
                    "2026-03-01",
                    "2026-06-10",
                ],
            }
        )
        cohorts = build_temporal_cohorts(frame)
        primary = cohorts["B_post_marker_current_temporal"]
        self.assertEqual(primary.development["sample_key"].tolist(), ["c", "d"])
        self.assertEqual(primary.test["sample_key"].tolist(), ["e"])
        pooled = cohorts["C_pooled_history_temporal"]
        self.assertEqual(pooled.development["sample_key"].tolist(), ["a", "c", "d"])

    def test_patient_disjoint_test_removes_seen_and_missing_groups(self) -> None:
        development = pd.DataFrame({"patient_cluster_id": ["p1", "p2"]})
        test = pd.DataFrame({"row": [1, 2, 3], "patient_cluster_id": ["p2", "p3", np.nan]})
        result = patient_disjoint_test(development, test)
        self.assertEqual(result["row"].tolist(), [2])

    def test_episode_first_requires_patient_and_flag(self) -> None:
        frame = pd.DataFrame(
            {
                "row": [1, 2, 3],
                "patient_cluster_id": ["p1", "p1", None],
                "episode_first_sample_flag": [True, False, True],
            }
        )
        self.assertEqual(episode_first_rows(frame)["row"].tolist(), [1])

    def test_rolling_origin_never_uses_future_test_rows(self) -> None:
        dates = pd.date_range("2025-07-01", "2026-02-28", freq="D")
        frame = pd.DataFrame(
            {
                "sample_key": [f"s{i}" for i in range(len(dates))],
                "date": dates,
                "y": np.arange(len(dates)) % 2,
            }
        )
        folds = rolling_origin_folds(frame)
        self.assertGreaterEqual(len(folds), 2)
        for train, validation, _ in folds:
            self.assertLess(train["date"].max(), validation["date"].min())
            self.assertLessEqual(validation["date"].max(), pd.Timestamp("2026-02-28"))


if __name__ == "__main__":
    unittest.main()
