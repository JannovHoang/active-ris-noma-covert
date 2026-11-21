"""Finite-family bounds, independent streams, and no-heldout-selection tests."""
from copy import deepcopy
from pathlib import Path
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.stats import binom

from ris_noma.config import load_config
from ris_noma.pso import PSOSettings
from ris_noma.validation import (family_constants, binomial_upper, fresh_rng,
                                 statistical_screen, select_on_validation)
import run_step3b

ROOT = Path(__file__).resolve().parents[1]


class TestValidationBounds(unittest.TestCase):
    def test_union_bound_failure_budget_matches_declared_alpha(self):
        n, J, K, alpha = 50000, 25, 2, .025
        b = family_constants(n, J, K, alpha)
        dep_failure = 2 * J * 2 * np.exp(-2 * n * b["cdf_epsilon_each"] ** 2)
        qos_failure = J * K * np.exp(-2 * n * b["rate_fraction_subtraction"] ** 2)
        outage_failure = J * b["outage_alpha_per_candidate"]
        self.assertAlmostEqual(dep_failure + qos_failure + outage_failure, alpha)

    def test_more_samples_narrow_bounds_and_more_proposals_widen_them(self):
        a = family_constants(10000, 5, 2, .05)
        b = family_constants(40000, 5, 2, .05)
        c = family_constants(10000, 50, 2, .05)
        self.assertAlmostEqual(a["dep_subtraction"] / b["dep_subtraction"], 2)
        self.assertGreater(c["dep_subtraction"], a["dep_subtraction"])
        self.assertGreater(c["rate_fraction_subtraction"], a["rate_fraction_subtraction"])

    def test_binomial_bound_inverts_tail_and_zero_events_is_not_zero_risk(self):
        n, alpha = 1000, .001
        for k in [0, 1, 20, 500, 999]:
            upper = binomial_upper(k, n, alpha)
            self.assertAlmostEqual(float(binom.cdf(k, n, upper)), alpha, places=10)
            self.assertGreater(upper, k / n)
        self.assertGreater(binomial_upper(0, n, alpha), 0)
        self.assertEqual(binomial_upper(n, n, alpha), 1)

    def test_invalid_counts_and_failure_budgets_rejected(self):
        for args in [(0, 1, 2, .05), (100, 0, 2, .05), (100, 1, 2, 0), (100, 1, 2, 1)]:
            with self.assertRaises(ValueError):
                family_constants(*args)
        for args in [(-1, 10, .05), (11, 10, .05), (0, 0, .05), (0, 10, 1)]:
            with self.assertRaises(ValueError):
                binomial_upper(*args)

    def test_fresh_streams_reproducible_and_distinct_from_old_evaluation(self):
        from ris_noma.random_streams import make_rng
        a = fresh_rng(7, 0, "validation", 1).normal(size=20)
        np.testing.assert_array_equal(a, fresh_rng(7, 0, "validation", 1).normal(size=20))
        self.assertFalse(np.array_equal(a, fresh_rng(7, 0, "heldout", 1).normal(size=20)))
        self.assertFalse(np.array_equal(a, make_rng(7, 0, "evaluation").normal(size=20)))

    def synthetic_screen(self, n=10000, xi=.75, public_rates=(.5, .8), fail_count=0):
        c = load_config(ROOT / "configs/legacy_physics_k2_pilot.json")
        summary = {"allocation_sic_order": {"mb": .2, "public": [.4, .4], "s": .5},
                   "outage_ceiling_margins_positive": True,
                   "constraints": {"feasible_under_gamma_sample_model": True}}
        failures = np.zeros(n, dtype=bool)
        failures[:fail_count] = True
        z = np.arange(1., n + 1.) / n
        samples = {"public_rates": np.tile(public_rates, (n, 1)), "bob_outage": failures,
                   "z0_willie": z, "z1_willie": z.copy()}
        return statistical_screen(c, summary, samples, [0, 1], xi, 25, .025)

    def test_identical_hypotheses_and_constant_rates_have_known_limits(self):
        r = self.synthetic_screen()
        self.assertEqual(r["dep_empirical_min"], 1)
        self.assertAlmostEqual(r["dep_uniform_lower"], 1 - r["constants"]["dep_subtraction"])
        self.assertTrue(r["accepted"])
        self.assertGreater(r["bob_outage_upper"], 0)

    def test_point_estimate_pass_does_not_override_insufficient_dep_bound(self):
        r = self.synthetic_screen(n=100, xi=.95)
        self.assertGreaterEqual(r["dep_empirical_min"], .95)
        self.assertFalse(r["checks"]["dep_uniform_lower_ge_xi"])
        self.assertFalse(r["accepted"])

    def test_qos_and_outage_bounds_each_can_reject_a_proposal(self):
        qos = self.synthetic_screen(public_rates=(.2, .8))
        self.assertFalse(qos["checks"]["public_mean_rate_lower_ge_qos"])
        self.assertFalse(qos["accepted"])
        outage = self.synthetic_screen(fail_count=1000)
        self.assertFalse(outage["checks"]["bob_outage_upper_le_limit"])
        self.assertFalse(outage["accepted"])


class TestSelection(unittest.TestCase):
    def proposals(self):
        return [{"proposal_id": name, "xi": .9, "label_feasible": True,
                 "label_metrics": {"bob_mean_rate_bit_s_hz": score}, "margin": margin, "seed": 1}
                for name, score, margin in [("a", 5., 0.), ("b", 4.8, .005), ("c", 4.5, .02)]]

    def test_highest_label_rate_among_validation_passes_is_selected(self):
        props = self.proposals()
        reports = [{"proposal_id": p["proposal_id"], "screen": {"accepted": p["proposal_id"] != "a"}} for p in props]
        selected = select_on_validation(props, reports, [.9])
        self.assertEqual(selected[0]["proposal_id"], "b")

    def test_label_failure_cannot_be_rescued_by_validation(self):
        props = self.proposals()
        props[0]["label_feasible"] = False
        reports = [{"proposal_id": p["proposal_id"], "screen": {"accepted": True}} for p in props]
        self.assertEqual(select_on_validation(props, reports, [.9])[0]["proposal_id"], "b")

    def test_no_pass_is_reported_without_claiming_infeasibility(self):
        props = self.proposals()
        reports = [{"proposal_id": p["proposal_id"], "screen": {"accepted": False}} for p in props]
        result = select_on_validation(props, reports, [.9])[0]
        self.assertIsNone(result["proposal_id"])
        self.assertEqual(result["status"], "no_candidate_passed_validation")

    def test_mismatched_or_duplicate_validation_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            select_on_validation(self.proposals(), [], [.9])
        with self.assertRaises(ValueError):
            select_on_validation(self.proposals(), [{"proposal_id": "a"}, {"proposal_id": "a"}], [.9])


class TestStep3bPipeline(unittest.TestCase):
    def setup_run(self):
        c = load_config(ROOT / "configs/legacy_physics_k2_pilot.json")
        c["requirements"].update(qos_public=[.01, .01], target_public=[.01, .01], target_bob=.01, xi=[.2])
        p = {"label_samples": 150, "validation_samples": 1500, "heldout_samples": 1500,
             "margins": [0., .02], "seeds": [1], "alpha": .05, "validation_seed": 1, "heldout_seed": 1}
        return c, PSOSettings(swarm_size=4, sweeps=2), p

    def test_selection_is_written_before_heldout_and_does_not_change_with_audit_seed(self):
        c, settings, policy = self.setup_run()
        original = run_step3b.build_bank
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "first"
            count = 0
            def checked_build(*args, **kwargs):
                nonlocal count
                count += 1
                if count == 3:
                    self.assertTrue((out / "selection_frozen.json").exists())
                return original(*args, **kwargs)
            with contextlib.redirect_stdout(io.StringIO()), patch.object(run_step3b, "build_bank", side_effect=checked_build):
                rows = run_step3b.run(c, settings, policy, [.2], out)
            self.assertIsNotNone(rows[0]["proposal_id"])
            self.assertEqual(count, 3)
            status = json.loads((out / "status.json").read_text())
            digest = hashlib.sha256((out / "selection_frozen.json").read_bytes()).hexdigest()
            self.assertEqual(status["selection_sha256_before_and_after_audit"], digest)
            second_policy = dict(policy, heldout_seed=999)
            with contextlib.redirect_stdout(io.StringIO()):
                second = run_step3b.run(c, settings, second_policy, [.2], Path(d) / "second", {.2: {"bob_rate_label": "12345"}})
            self.assertEqual(rows[0]["proposal_id"], second[0]["proposal_id"])
            self.assertEqual(rows[0]["mb"], second[0]["mb"])
            self.assertFalse(rows[0]["heldout_used_for_selection"])

    def test_empty_validation_selection_does_not_generate_heldout(self):
        c, settings, policy = self.setup_run()
        # With only two validation samples the uniform DEP lower bound is zero.
        policy["validation_samples"] = 2
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            rows = run_step3b.run(c, settings, policy, [.2], d)
            self.assertIsNone(rows[0]["proposal_id"])
            status = json.loads((Path(d) / "status.json").read_text())
            self.assertFalse(status["heldout_generated"])

    def test_nonempty_output_is_protected(self):
        c, settings, policy = self.setup_run()
        with tempfile.TemporaryDirectory() as d:
            marker = Path(d) / "existing.txt"
            marker.write_text("keep")
            with self.assertRaisesRegex(ValueError, "not empty"):
                run_step3b.run(c, settings, policy, [.2], d)
            self.assertEqual(marker.read_text(), "keep")

    def test_invalid_gamma_targets_are_explicitly_skipped(self):
        _, _, policy = self.setup_run()
        policy["margins"] = [0., .02]
        specs, skipped = run_step3b.make_specs([.99], policy)
        self.assertEqual(len(specs), 1)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["reason"], "xi_plus_margin_ge_1")


if __name__ == "__main__":
    unittest.main()
