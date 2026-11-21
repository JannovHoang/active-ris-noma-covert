"""Regression checks for multi-power identities and paired streams in step 6c."""
import unittest

import numpy as np

try:
    from ris_noma.expanded_benchmark import paired_rng, run_key
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    paired_rng = None


@unittest.skipIf(paired_rng is None, "Install the step 6b environment")
class TestExpandedBenchmark(unittest.TestCase):
    def test_same_layout_and_dep_at_different_powers_have_distinct_records_and_rng(self):
        low = {"layout_id": 209, "total_power_dbm": 30., "power_index": 0, "xi_index": 4, "xi": .95}
        high = dict(low, total_power_dbm=40., power_index=2)
        self.assertNotEqual(run_key(low, 101, "pso_short"), run_key(high, 101, "pso_short"))
        expected = paired_rng(7, low, "optimizer", 101).uniform(size=(30, 3))
        paired_rng(7, low, "warm_jitter", 101).normal(size=(5, 3))
        np.testing.assert_array_equal(paired_rng(7, low, "optimizer", 101).uniform(size=(30, 3)), expected)
        self.assertFalse(np.array_equal(paired_rng(7, high, "optimizer", 101).uniform(size=(30, 3)), expected))
