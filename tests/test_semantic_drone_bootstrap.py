"""Tests for paired raster aggregation of Semantic Drone confirmation outputs."""

from __future__ import annotations

import unittest

from p1eval.semantic_drone_bootstrap import (
    aggregate_grass_counts,
    exclude_identifiers,
    paired_raster_bootstrap_difference,
    paired_raster_bootstrap_mean_difference,
)


class SemanticDroneBootstrapTest(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = {
            "000": {"true_positive": 2, "false_positive": 1, "false_negative": 3},
            "001": {"true_positive": 1, "false_positive": 2, "false_negative": 4},
            "002": {"true_positive": 3, "false_positive": 1, "false_negative": 2},
        }
        self.model = {
            "000": {"true_positive": 4, "false_positive": 1, "false_negative": 1},
            "001": {"true_positive": 4, "false_positive": 1, "false_negative": 1},
            "002": {"true_positive": 5, "false_positive": 0, "false_negative": 0},
        }

    def test_aggregate_uses_global_counts_not_mean_per_raster_iou(self) -> None:
        result = aggregate_grass_counts(self.baseline)
        self.assertEqual(result["true_positive"], 6)
        self.assertEqual(result["false_positive"], 4)
        self.assertEqual(result["false_negative"], 9)
        self.assertAlmostEqual(result["grass_iou"], 6 / 19)

    def test_paired_bootstrap_is_deterministic_and_positive_for_better_model(self) -> None:
        first = paired_raster_bootstrap_difference(self.baseline, self.model, replicates=200, seed=19)
        second = paired_raster_bootstrap_difference(self.baseline, self.model, replicates=200, seed=19)
        self.assertEqual(first, second)
        self.assertEqual(first["resampling_unit"], "semantic_drone_raster")
        self.assertGreater(first["point_difference"], 0.0)
        self.assertGreater(first["confidence_interval"]["lower"], 0.0)

    def test_rejects_unpaired_identifiers(self) -> None:
        with self.assertRaisesRegex(ValueError, "identifier sets"):
            paired_raster_bootstrap_difference(self.baseline, {"000": self.model["000"]}, replicates=200)

    def test_mean_difference_is_deterministic_across_seed_pairs(self) -> None:
        first = paired_raster_bootstrap_mean_difference(
            [self.baseline, self.baseline], [self.model, self.model], replicates=200, seed=19
        )
        second = paired_raster_bootstrap_mean_difference(
            [self.baseline, self.baseline], [self.model, self.model], replicates=200, seed=19
        )
        self.assertEqual(first, second)
        self.assertEqual(first["model_seed_count"], 2)
        self.assertGreater(first["confidence_interval"]["lower"], 0.0)

    def test_excludes_declared_identifiers_without_mutating_source(self) -> None:
        retained = exclude_identifiers(self.baseline, ("000", "002"))
        self.assertEqual(tuple(sorted(retained)), ("001",))
        self.assertEqual(tuple(sorted(self.baseline)), ("000", "001", "002"))
        with self.assertRaisesRegex(ValueError, "absent"):
            exclude_identifiers(self.baseline, ("999",))


if __name__ == "__main__":
    unittest.main()
