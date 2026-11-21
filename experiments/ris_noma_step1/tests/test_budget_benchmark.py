"""Regression tests for PSO snapshots and layout-level budget pairing."""
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest

import numpy as np

from ris_noma.budget_analysis import report
from ris_noma.pso import Outcome, PSOSettings, optimize

try:
    from ris_noma.budget_benchmark import capture_budgets
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    capture_budgets = None


class TestBudgetCapture(unittest.TestCase):
    @unittest.skipIf(capture_budgets is None, "Install the step 6b environment")
    def test_uses_one_trajectory_and_preserves_first_and_last_best(self):
        settings = PSOSettings(swarm_size=4, sweeps=5)
        seen = []

        def objective(z):
            seen.append(z.copy())
            return Outcome(float(np.round(z.sum(), 1)), 0., True)

        budgets = [4, 8, 20]
        snapshots = capture_budgets(objective, settings, np.random.default_rng(5), None, budgets,
                                    time.perf_counter(), time.process_time())
        self.assertEqual(len(seen), 20)
        self.assertEqual([row["budget"] for row in snapshots], budgets)
        np.testing.assert_array_equal(snapshots[0]["z"], max(seen[:4], key=lambda z: np.round(z.sum(), 1)))
        result = optimize(objective, 3, settings, np.random.default_rng(5))
        np.testing.assert_array_equal(snapshots[-1]["z"], result.position)


class TestBudgetPairing(unittest.TestCase):
    def test_deltas_use_matched_layouts_and_cross_budget_references(self):
        plan = {"layout_ids": [300, 301], "budgets": [30, 60], "primary_budget": 30, "cross_budget_reference": 60,
                "power_dbm": [40.], "xi": [.9], "search_config": {"pso_seeds": [101]},
                "audit_samples_per_layout": 1000, "bootstrap_repetitions": 1000, "streams": {"bootstrap": 27}}
        rows = []
        for lid in plan["layout_ids"]:
            for method in ("pso_short", "gnn_pso_short"):
                for budget in plan["budgets"]:
                    passed = budget == 60 or (lid == 300 and method == "gnn_pso_short")
                    rows.append({"layout_id": lid, "total_power_dbm": 40., "xi": .9, "pso_seed": 101,
                                 "method": method, "budget": budget, "audit_pass": passed, "search_feasible": True,
                                 "bob_rate": 4., "gated_bob_rate": 4. if passed else 0.,
                                 "search_wall_seconds": .01 * budget, "search_cpu_seconds": .01 * budget})
        with TemporaryDirectory() as folder:
            summary = report(Path(folder), plan, rows)
        self.assertEqual(summary["primary"]["pass_delta_pp"], 50.)
        self.assertEqual(summary["primary"]["pass_delta_pp_ci_high"], 100.)
        self.assertEqual(summary["cross_budget"]["pass_delta_pp"], -50.)
        self.assertEqual(summary["paired_trajectories"], 2)
