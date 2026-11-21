"""Regression for the multi-power prediction join in step 6b."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ris_noma.expanded_learning_physics import PREDICTION_FIELDS, load_prediction
from ris_noma.learning_data import TARGET_KEY, sha256


class TestExpandedLearning(unittest.TestCase):
    def test_wrong_power_rejected_even_if_layout_xi_and_reference_match(self):
        allocations = np.array([[.2, .5, .3, .5], [.2, .5, .3, .5]])
        data = {
            "layout_id": np.array([209, 209]),
            "total_power_dbm": np.array([30., 40.]),
            "xi": np.array([.85, .85]),
            "sic_order_original_ids": np.array([[0, 1], [0, 1]]),
            TARGET_KEY: allocations,
        }
        arrays = {key: data[key] for key in PREDICTION_FIELDS}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "frozen.npz"
            np.savez_compressed(path, prediction=allocations, reference=allocations, **arrays)
            np.testing.assert_array_equal(load_prediction(path, sha256(path), data), allocations)

            arrays["total_power_dbm"] = arrays["total_power_dbm"][::-1]
            np.savez_compressed(path, prediction=allocations, reference=allocations, **arrays)
            with self.assertRaisesRegex(ValueError, "Prediction identity mismatch.*total_power_dbm"):
                load_prediction(path, sha256(path), data)
