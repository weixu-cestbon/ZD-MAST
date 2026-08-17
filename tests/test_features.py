from __future__ import annotations

import unittest

import numpy as np

from zd_mast.features import (
    FeatureSchema,
    aggregate_replicates,
    bin_spectrum,
    dense_profile_peak_presence,
    peak_presence,
)


class FeatureTests(unittest.TestCase):
    def test_schema_has_6000_features(self) -> None:
        self.assertEqual(FeatureSchema().n_features, 6000)

    def test_bin_uses_max_and_excludes_upper_boundary(self) -> None:
        mz = np.array([1999.0, 2000.1, 2001.0, 2003.0, 19999.9, 20000.0])
        intensity = np.array([99.0, 4.0, 9.0, 16.0, 25.0, 99.0])
        vector = bin_spectrum(mz, intensity)
        self.assertEqual(vector.shape, (6000,))
        self.assertGreater(vector[0], 0)
        self.assertGreater(vector[1], vector[0])
        self.assertGreater(vector[-1], 0)
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=6)

    def test_aggregate_replicates_normalizes(self) -> None:
        result = aggregate_replicates(np.array([[1.0, 0.0], [0.0, 1.0]]))
        self.assertTrue(np.allclose(result, [2**-0.5, 2**-0.5]))

    def test_peak_presence_is_binary(self) -> None:
        result = peak_presence(np.array([[0.0, 0.2, -0.0]]))
        self.assertEqual(result.tolist(), [[0, 1, 0]])

    def test_dense_profile_presence_does_not_mark_positive_baseline(self) -> None:
        mz = np.linspace(2000.0, 2200.0, 1000, endpoint=False)
        intensity = np.full_like(mz, 10.0)
        intensity += 100.0 * np.exp(-0.5 * ((mz - 2100.0) / 0.8) ** 2)
        result = dense_profile_peak_presence(mz, intensity)
        self.assertEqual(result.dtype, np.uint8)
        self.assertGreater(int(result.sum()), 0)
        self.assertLess(int(result.sum()), 10)


if __name__ == "__main__":
    unittest.main()
