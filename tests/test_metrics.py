from __future__ import annotations

import unittest

import numpy as np

from zd_mast.metrics import threshold_at_min_sensitivity, threshold_at_min_specificity, youden_threshold


class MetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.y = np.array([0, 0, 0, 1, 1, 1])
        self.p = np.array([0.05, 0.20, 0.40, 0.60, 0.80, 0.95])

    def test_sensitivity_threshold_is_finite_and_specific(self) -> None:
        point = threshold_at_min_sensitivity(self.y, self.p, 0.90)
        self.assertEqual(point.threshold, 0.60)
        self.assertEqual(point.sensitivity, 1.0)
        self.assertEqual(point.specificity, 1.0)

    def test_specificity_threshold_does_not_return_infinity(self) -> None:
        point = threshold_at_min_specificity(self.y, self.p, 0.90)
        self.assertTrue(np.isfinite(point.threshold))
        self.assertEqual(point.threshold, 0.60)
        self.assertEqual(point.specificity, 1.0)

    def test_youden_threshold(self) -> None:
        point = youden_threshold(self.y, self.p)
        self.assertEqual(point.threshold, 0.60)


if __name__ == "__main__":
    unittest.main()
