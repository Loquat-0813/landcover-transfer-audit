"""Tests for the declared supported-leaf bootstrap summary."""

from __future__ import annotations

import unittest

import numpy as np

from p1eval.supported_leaf_bootstrap import (
    paired_city_bootstrap_mean_supported_leaf_difference,
    supported_leaf_mean_iou,
)


class SupportedLeafBootstrapTest(unittest.TestCase):
    def setUp(self) -> None:
        self.names = ("built", "herbaceous", "water")
        self.baseline = {
            "city_a": np.asarray([[8, 1, 1], [0, 8, 2], [2, 1, 7]]),
            "city_b": np.asarray([[6, 2, 2], [0, 6, 4], [2, 2, 6]]),
        }
        self.model = {
            "city_a": np.asarray([[9, 0, 1], [0, 9, 1], [1, 0, 9]]),
            "city_b": np.asarray([[7, 1, 2], [0, 7, 3], [1, 1, 8]]),
        }

    def test_selected_mean_keeps_errors_into_excluded_leaf(self) -> None:
        value = supported_leaf_mean_iou(self.baseline["city_a"], self.names, ("built", "water"))
        expected_built = 8 / (10 + 10 - 8)
        expected_water = 7 / (10 + 10 - 7)
        self.assertAlmostEqual(value, (expected_built + expected_water) / 2)

    def test_paired_bootstrap_is_deterministic(self) -> None:
        first = paired_city_bootstrap_mean_supported_leaf_difference(
            [self.baseline], [self.model], self.names, ("built", "water"), replicates=200, seed=19
        )
        second = paired_city_bootstrap_mean_supported_leaf_difference(
            [self.baseline], [self.model], self.names, ("built", "water"), replicates=200, seed=19
        )
        self.assertEqual(first, second)
        self.assertGreater(first["point_difference"], 0.0)
        self.assertGreater(first["confidence_interval"]["lower"], 0.0)

    def test_rejects_unknown_supported_leaf(self) -> None:
        with self.assertRaisesRegex(ValueError, "absent"):
            supported_leaf_mean_iou(self.baseline["city_a"], self.names, ("unknown",))


if __name__ == "__main__":
    unittest.main()
