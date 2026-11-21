"""Regression checks for multi-power identity and the frozen audit boundary."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ris_noma.config import load_config
from ris_noma.expanded_data import check_run, generate
from ris_noma.expanded_scenarios import ROOT, make_plan
from ris_noma.graph_data import GLOBAL_FEATURES
from ris_noma.learning_data import read_json, sha256
from ris_noma.pso import PSOSettings


def small_plan():
    physical = load_config(ROOT / "configs/legacy_physics_k2_pilot.json")
    physical["positions"]["public_center"] = [130., 4.]
    physical["public_half_width_m"] = 2.
    physical["requirements"]["xi"] = [.15, .25]
    experiment = read_json(ROOT / "configs/dataset_pilot_6a.json")
    experiment.update(
        splits={"train": [200], "validation": [201], "test": [202]},
        total_power_dbm=[30., 40.], pso_seeds=[1],
        label_samples=256, validation_samples=2000, heldout_samples=2000,
        position_half_width_m={"bob": 1., "willie": 1.},
    )
    return make_plan(physical, PSOSettings(swarm_size=6, sweeps=3), experiment)


class TestExpandedData(unittest.TestCase):
    def test_power_is_part_of_case_identity_and_audit_resume_reuses_frozen_labels(self):
        plan = small_plan()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            output = Path(tmp) / "run"
            with patch("ris_noma.expanded_data.audit_layout", side_effect=RuntimeError("interrupted audit")):
                with self.assertRaisesRegex(RuntimeError, "interrupted audit"):
                    generate(plan, output)
            frozen = output / "selections_frozen.json"
            self.assertTrue(frozen.is_file())
            self.assertEqual(len(read_json(frozen)["layouts"]), 3)
            digest = sha256(frozen)
            self.assertFalse((output / "run.lock").exists())
            with patch("ris_noma.expanded_data.search_layout", side_effect=AssertionError("reran PSO")), \
                    patch("ris_noma.expanded_data.select_layout", side_effect=AssertionError("reselected labels")):
                summary = generate(plan, output, resume=True)
            self.assertEqual(digest, sha256(frozen))
            self.assertEqual(summary["cases"]["labeled"], 12)
            self.assertEqual(summary["pso_evaluations"], 216)
            self.assertEqual(check_run(output), summary)
            index = GLOBAL_FEATURES.index("log10_total_power_w")
            for role, layout_id in [("train", 200), ("validation", 201), ("test", 202)]:
                with np.load(output / "graphs" / f"{role}.npz", allow_pickle=False) as data:
                    self.assertEqual(set(data["layout_id"]), {layout_id})
                    self.assertEqual(len(set(zip(data["total_power_dbm"], data["xi"]))), 4)
                    np.testing.assert_allclose(data["global_raw"][:, index], [0., 0., 1., 1.])
                    np.testing.assert_array_equal(data["node_raw"][0], data["node_raw"][2])
            with self.assertRaisesRegex(ValueError, "Nonempty"):
                generate(plan, output)
            self.assertEqual(check_run(output), summary)
            self.assertFalse((output / "run.lock").exists())

    def test_changed_committed_result_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            output = Path(tmp) / "run"
            generate(small_plan(), output)
            path = output / "cases.csv"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "result changed: cases.csv"):
                check_run(output)
