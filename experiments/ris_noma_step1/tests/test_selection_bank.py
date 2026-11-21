"""6g preserves the optimizer, freezes finite pools, and counts abstentions."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
    from ris_noma.candidate_archive import CandidateArchive, capture_candidates, position_key
    from ris_noma.selection_bank import (STREAMS, audit_layout, choose_candidate, freeze_margin,
                                         rows_per_case, select_layout, validate_config)
    from ris_noma.learning_data import read_json
    from ris_noma.pso import Outcome, PSOSettings, optimize
    from ris_noma.graph_data import original_allocation
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

ROOT = Path(__file__).resolve().parents[1]


def outcome(position, target=.8):
    dep = float(1. - .3 * position[0])
    violation = max(0., target - dep)
    return Outcome(float(position.sum()), violation, violation == 0.,
                   {"dep_min": dep, "normalized_violations": {"covertness": violation, "qos": 0., "outage": 0., "layout": 0.}})


@unittest.skipIf(torch is None, "Install the project PyTorch environment")
class TestSelectionBank(unittest.TestCase):
    def test_archive_does_not_change_pso_trajectory_or_budget(self):
        bounds = {"mb": [.05, .49], "s": [.05, .95]}
        settings = PSOSettings(swarm_size=6, sweeps=2, initial_velocity_scale=.1)
        initial = np.array([[.7, .4, .3]])
        expected_points, actual_points = [], []
        def reference(z):
            expected_points.append(z.copy())
            return outcome(z)
        expected = optimize(reference, 3, settings, np.random.default_rng(12), initial)
        def observed(z):
            actual_points.append(z.copy())
            return outcome(z)
        points = capture_candidates(observed, settings, np.random.default_rng(12), initial, [6, 12],
                                    .8, [.0075, .015], np.array([1, 0]), bounds)
        np.testing.assert_array_equal(actual_points, expected_points)
        self.assertEqual(len(actual_points), 12)
        final = next(c for c in points[-1]["candidates"] if c["candidate_id"] == points[-1]["raw_candidate_id"])
        y = np.array(final["allocation"])
        _, z = original_allocation({"mb": y[0], "public": y[1:3][[1, 0]], "s": y[3]}, np.array([1, 0]), bounds)
        np.testing.assert_allclose(z, expected.position, atol=1e-14)
        self.assertEqual([r["budget"] for r in points], [6, 12])
        self.assertTrue(all(len(r["candidates"]) <= 5 for r in points))

    def test_poor_gnn_point_is_preserved_without_displacing_raw_best(self):
        gnn = np.array([.99, 0., 0.])
        archive = CandidateArchive(.8, [.0075, .015], protected_position=gnn)
        archive.observe(gnn, outcome(gnn))
        for a in np.linspace(.1, .6, 12):
            z = np.array([a, 1., 1.])
            archive.observe(z, outcome(z))
        pool = archive.pool()
        self.assertEqual(len(pool), 5)
        self.assertIn(position_key(gnn), [r["key"] for r in pool])
        self.assertIn(archive.top[0]["key"], [r["key"] for r in pool])
        self.assertEqual(len({r["key"] for r in pool}), len(pool))

    def test_archive_safest_candidate_must_pass_other_search_constraints(self):
        archive = CandidateArchive(.8, [.0075, .015])
        archive.observe(np.array([.4, .3, .2]), outcome(np.array([.4, .3, .2])))
        bad = Outcome(5., .1, False, {"dep_min": .999, "normalized_violations": {"covertness": 0., "qos": .1, "outage": 0., "layout": 0.}})
        archive.observe(np.zeros(3), bad)
        self.assertAlmostEqual(archive.safest["outcome"].metrics["dep_min"], .88)

    def test_selection_rejects_faster_failed_candidates_without_repair(self):
        rows = [{"candidate_id": "a", "audit_pass": False, "bob_rate": 9., "dep_lower": .799},
                {"candidate_id": "b", "audit_pass": True, "bob_rate": 4., "dep_lower": .81},
                {"candidate_id": "c", "audit_pass": True, "bob_rate": 4.1, "dep_lower": .805}]
        before = deepcopy(rows)
        self.assertEqual(choose_candidate(rows)["candidate_id"], "c")
        self.assertEqual(rows, before)
        self.assertIsNone(choose_candidate([rows[0]]))
        with self.assertRaises(ValueError):
            choose_candidate([rows[0], rows[0]])

    def test_configuration_counts_and_streams_are_predeclared(self):
        c = read_json(ROOT / "configs/selection_bank_6g.json")
        validate_config(c, 30, [.75, .8, .85, .9, .95])
        plan = {**c, "search_config": {"pso_seeds": [101, 202, 303]}}
        self.assertEqual(rows_per_case(plan, "test"), 44)
        self.assertEqual(rows_per_case(plan, "validation"), 38)
        self.assertEqual(len(set(STREAMS.values())), len(STREAMS))
        self.assertTrue(min(STREAMS.values()) > 42)
        for key, value in (("test_layout_ids", [500, 501]), ("budgets", [30, 31]),
                           ("candidate_limit", 4), ("archive_slacks", [.01]), ("search_margins", [.05])):
            with self.assertRaises(ValueError):
                validate_config(dict(c, **{key: value}), 30, [.75, .8, .85, .9, .95])

    def test_abstention_is_kept_and_never_forward_evaluated_as_an_allocation(self):
        plan = {"selection_samples": 1000, "audit_samples": 1000, "test_selection_family_size": 5,
                "test_audit_family_size": 2, "test_alpha_per_family": .01}
        candidate = {"candidate_id": "gnn", "allocation": [.2, .3, .5, .4]}
        pool = {"layout_id": 600, "total_power_dbm": 30., "xi": .8, "search_margin": .02,
                "branch": "gnn", "budget": 0, "pso_seed": None, "search_wall_seconds": .01,
                "search_bank_seconds": .1, "objective_cache_seconds": 0.,
                "raw_candidate_id": "gnn", "candidates": [candidate]}
        search = {"order": [0, 1], "pools": [pool]}
        failed = {**candidate, "audit_pass": False, "bob_rate": 4., "dep_lower": .79}
        context = ({}, object(), object())
        with patch("ris_noma.selection_bank.channel_context", return_value=context), \
             patch("ris_noma.selection_bank.score_cases", return_value=([failed], 1)):
            decisions = select_layout({"layout_id": 600}, plan, {}, search, "test", "candidatehash")
        raw, screened = decisions["rows"]
        self.assertFalse(raw["abstained"])
        self.assertTrue(screened["abstained"])
        self.assertIsNone(screened["allocation"])
        def score(base, layout, bank, order, rows, size, alpha):
            self.assertEqual(len(rows), 1)
            self.assertIsNotNone(rows[0]["allocation"])
            return [{**rows[0], "audit_pass": True, "bob_rate": 4., "gated_bob_rate": 4., "failed_checks": []}], 1
        with patch("ris_noma.selection_bank.channel_context", return_value=context), \
             patch("ris_noma.selection_bank.score_cases", side_effect=score):
            result = audit_layout({"layout_id": 600}, plan, {}, decisions, [0, 1], "test", "decisionhash")
        self.assertEqual(len(result["rows"]), 2)
        abstained = next(r for r in result["rows"] if r["abstained"])
        self.assertFalse(abstained["audit_pass"])
        self.assertEqual(abstained["gated_bob_rate"], 0.)
        self.assertEqual(result["unique_audit_evaluations"], 1)

    def test_margin_is_selected_on_pooled_validation_audits_only(self):
        rows = []
        for margin, passed, rate in ((.02, False, 9.), (.03, True, 4.)):
            for method in ("cold_screened", "warm_screened"):
                rows.append({"search_margin": margin, "budget": 60, "method": method,
                             "audit_pass": passed, "gated_bob_rate": rate if passed else 0., "abstained": not passed})
        with TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "validation_audit_frozen.json").write_text('{}')
            result = freeze_margin(path, {"search_margins": [.02, .03], "primary_budget": 60}, rows)
            self.assertEqual(result["chosen_search_margin"], .03)
            self.assertFalse(result["test_used_for_selection"])


if __name__ == "__main__":
    unittest.main()
