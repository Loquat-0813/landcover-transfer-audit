"""Tests for the deterministic source-order invariant used by B0/B1."""

from __future__ import annotations

import unittest

from p1train.fixed_budget import CyclingPermutation, loveda_positions_for_update


class FixedBudgetTest(unittest.TestCase):
    def test_cycle_is_complete_without_replacement(self) -> None:
        sampler = CyclingPermutation(length=11, seed=19, stream=3)
        first = [sampler.at(position) for position in range(11)]
        second = [sampler.at(position) for position in range(11, 22)]
        self.assertEqual(sorted(first), list(range(11)))
        self.assertEqual(sorted(second), list(range(11)))
        self.assertNotEqual(first, second)

    def test_seed_and_stream_define_stable_independent_orders(self) -> None:
        first = [CyclingPermutation(17, 37, 0).at(position) for position in range(17)]
        repeated = [CyclingPermutation(17, 37, 0).at(position) for position in range(17)]
        other_stream = [CyclingPermutation(17, 37, 1).at(position) for position in range(17)]
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, other_stream)

    def test_source_exposure_positions_distinguish_primary_and_half_control(self) -> None:
        self.assertEqual(loveda_positions_for_update("b0", 7), (14, 15))
        self.assertEqual(loveda_positions_for_update("b0_half", 7), (7,))
        self.assertEqual(loveda_positions_for_update("b0_half_batch2", 7), (7,))
        self.assertEqual(loveda_positions_for_update("b1", 7), (7,))
        with self.assertRaises(ValueError):
            loveda_positions_for_update("unknown", 0)


if __name__ == "__main__":
    unittest.main()
