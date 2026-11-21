"""Nested target selection, immutable sources and fresh heldout auditing."""
from copy import deepcopy
from pathlib import Path
import contextlib
import io
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from ris_noma.config import load_config
from ris_noma.pso import PSOSettings
from ris_noma.validation import family_constants, binomial_upper
from ris_noma.cross_xi import target_checks, select_across_xi, verify_validation_records
import run_step3b
import run_step4a
import run_step4b

ROOT = Path(__file__).resolve().parents[1]


class TestCrossTargetSelection(unittest.TestCase):
    def setUp(self):
        self.c = load_config(ROOT / "configs/legacy_physics_k2_pilot.json")
        self.order = [1, 0]

    def pair(self, name="p1", source=.9, lower=.88, rate=4.):
        allocation = {"mb": .2, "public": [.4, .4], "s": .5}
        p = {"proposal_id": name, "xi": source, "margin": .02, "gamma_target": source + .02,
             "seed": 1, "label_feasible": True,
             "label_metrics": {"dep_min": source + .02, "bob_mean_rate_bit_s_hz": rate,
                               "allocation_sic_order": deepcopy(allocation)}}
        a = {"proposal_id": name, "metrics": {"allocation_sic_order": allocation,
             "dep_min": max(.94, source + .02), "public_mean_rate_bit_s_hz": [.5, .6], "bob_outage_estimate": 0.,
             "outage_ceiling_margins_positive": True,
             "constraints": {"checks": {"layout_bob_mean_gain_rule": True}}},
             "screen": {"accepted": False, "dep_empirical_min": lower + .01,
                        "dep_uniform_lower": lower, "public_rate_lower_sic_order": [.45, .55],
                        "bob_outage_upper": .001}}
        return p, a

    def test_high_target_proposal_can_pass_lower_target_despite_old_false_flag(self):
        p, a = self.pair()
        result = select_across_xi(self.c, self.order, [p], [a], [.75, .85, .9])
        self.assertEqual([r["proposal_id"] for r in result], ["p1", "p1", None])
        self.assertAlmostEqual(result[0]["effective_gamma_margin"], .17)
        self.assertEqual(result[0]["source_margin"], .02)

    def test_loosening_xi_never_overrides_other_requirements(self):
        p, a = self.pair()
        changes = [("screen", "public_rate_lower_sic_order", [.1, .55]),
                   ("screen", "bob_outage_upper", .2),
                   ("metrics", "outage_ceiling_margins_positive", False)]
        for section, key, value in changes:
            b = deepcopy(a);b[section][key] = value
            self.assertIsNone(select_across_xi(self.c, self.order, [p], [b], [.75])[0]["proposal_id"])
        q = deepcopy(p);q["label_feasible"] = False
        self.assertFalse(all(target_checks(self.c, self.order, q, a, .75).values()))

    def test_best_label_rate_is_selected_and_nonincreasing_across_targets(self):
        pairs = [self.pair("fast", .8, .78, 5.), self.pair("middle", .9, .88, 4.),
                 self.pair("strict", .95, .96, 3.)]
        ps, assessments = zip(*pairs)
        result = select_across_xi(self.c, self.order, ps, assessments, [.95, .75, .85])
        self.assertEqual([r["proposal_id"] for r in result], ["strict", "fast", "middle"])
        old = select_across_xi(self.c, self.order, [ps[2]], [assessments[2]], [.75])[0]
        self.assertEqual(old["proposal_id"], "strict")

    def test_original_pass_flag_cannot_override_a_failed_numeric_bound(self):
        p, a = self.pair(lower=.74)
        a["screen"]["accepted"] = True
        self.assertIsNone(select_across_xi(self.c, self.order, [p], [a], [.75])[0]["proposal_id"])
        self.assertTrue(all(target_checks(self.c, self.order, p, a, .75, bounded=False).values()))

    def test_duplicate_or_unmatched_proposal_ids_are_rejected(self):
        p, a = self.pair()
        for ps, assessments in [([p, p], [a]), ([p], []), ([p], [a, a])]:
            with self.assertRaises(ValueError):
                select_across_xi(self.c, self.order, ps, assessments, [.75])

    def test_saved_validation_bounds_and_allocations_must_reconcile(self):
        p, a = self.pair()
        n, alpha = 10000, .025
        constants = family_constants(n, 1, 2, alpha)
        mp = np.array([.4, .4]);A = .2 + np.cumsum(mp[::-1])[::-1] - mp + self.c["kappa"]**2
        a["metrics"].update(n_fading=n, bob_outage_count=0)
        a["screen"].update(constants=constants,
                           public_rate_lower_sic_order=(np.array([.5, .6]) - np.log2(1 + mp/A)*constants["rate_fraction_subtraction"]).tolist(),
                           dep_uniform_lower=a["screen"]["dep_empirical_min"] - constants["dep_subtraction"],
                           bob_outage_upper=binomial_upper(0, n, constants["outage_alpha_per_candidate"]))
        verify_validation_records(self.c, self.order, [p], [a], n, 1, alpha)
        bad = deepcopy(a);bad["screen"]["dep_uniform_lower"] += .001
        with self.assertRaisesRegex(ValueError, "DEP confidence"):
            verify_validation_records(self.c, self.order, [p], [bad], n, 1, alpha)
        bad = deepcopy(a);bad["metrics"]["allocation_sic_order"]["s"] = .6
        with self.assertRaisesRegex(ValueError, "allocations differ"):
            verify_validation_records(self.c, self.order, [p], [bad], n, 1, alpha)


class TestCrossTargetRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)
        cls.source = cls.base / "source"
        c = load_config(ROOT / "configs/legacy_physics_k2_pilot.json")
        c["requirements"].update(xi=[.2, .4], qos_public=[.01, .01], target_public=[.01, .01], target_bob=.01)
        policy = run_step3b.load_policy(ROOT / "configs/validation_pilot.json")
        policy.update(label_samples=150, validation_samples=1500, heldout_samples=1500, margins=[0., .02])
        study = {"layout_ids": [1], "seeds": [1], "study_alpha": .05}
        with contextlib.redirect_stdout(io.StringIO()):
            run_step4a.run_study(c, PSOSettings(swarm_size=4, sweeps=2), policy, study, cls.source)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_new_audit_is_frozen_first_evaluates_shared_allocation_once_and_preserves_source(self):
        before = {str(p.relative_to(self.source)): run_step4a.sha(p) for p in self.source.rglob("*") if p.is_file()}
        output = self.base / "new_audit"
        real_build = run_step4b.build_bank
        def checked_build(*args):
            self.assertTrue((output / "layout_000001/selection_frozen.json").exists())
            return real_build(*args)
        with contextlib.redirect_stdout(io.StringIO()), patch.object(run_step4b, "build_bank", side_effect=checked_build), \
                patch.object(run_step4b, "assess_proposal", wraps=run_step4b.assess_proposal) as assess:
            status, summaries = run_step4b.run(self.source, output)
        rows = run_step4a.read_json(output / "layout_000001/selected_by_xi.json")
        self.assertTrue(status["study_complete"])
        self.assertEqual(status["new_pso_candidate_evaluations"], 0)
        self.assertEqual([r["validation_selected_count"] for r in summaries], [1, 1])
        self.assertEqual(assess.call_count, len({r["proposal_id"] for r in rows}))
        self.assertEqual(assess.call_count, 1)
        self.assertTrue(all(r["paired_label_rate_gain"] >= -1e-10 for r in rows if r["paired_label_rate_gain"] is not None))
        self.assertEqual(before, {str(p.relative_to(self.source)): run_step4a.sha(p) for p in self.source.rglob("*") if p.is_file()})

    def test_resume_skips_completed_audits_and_rejects_changed_seed(self):
        output = self.base / "resumed_audit"
        with contextlib.redirect_stdout(io.StringIO()):
            run_step4b.run(self.source, output)
            with patch.object(run_step4b, "build_bank", side_effect=AssertionError("Must skip completed audit")):
                status, _ = run_step4b.run(self.source, output, resume=True)
        self.assertTrue(status["study_complete"])
        with self.assertRaisesRegex(ValueError, "source/code/seed changed"):
            run_step4b.run(self.source, output, heldout_seed=3, resume=True)

    def test_failed_fresh_audit_is_reported_without_replacing_selected_candidate(self):
        real_assess = run_step4b.assess_proposal
        def fail_dep(*args):
            result = real_assess(*args)
            result["screen"]["dep_uniform_lower"] = 0.
            return result
        output = self.base / "failed_audit"
        with contextlib.redirect_stdout(io.StringIO()), patch.object(run_step4b, "assess_proposal", side_effect=fail_dep):
            _, summaries = run_step4b.run(self.source, output)
        chosen = run_step4a.read_json(output / "layout_000001/selection_frozen.json")["selection"]
        rows = run_step4a.read_json(output / "layout_000001/selected_by_xi.json")
        self.assertEqual([r["proposal_id"] for r in rows], [r["proposal_id"] for r in chosen])
        self.assertTrue(all(r["status"] == "selected_on_validation_audit_review" for r in rows))
        self.assertEqual([s["heldout_review_count"] for s in summaries], [1, 1])

    def test_old_heldout_seed_and_nested_output_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "different from the source"):
            run_step4b.source_plan(self.source, 1)
        with self.assertRaisesRegex(ValueError, "non-nested"):
            run_step4b.run(self.source, self.source / "bad_output")

    def test_incomplete_source_is_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "incomplete";source.mkdir()
            plan = run_step4a.read_json(self.source / "study_plan.json")
            status = run_step4a.read_json(self.source / "study_status.json")
            status["study_complete"] = False
            run_step3b.write_json(source / "study_plan.json", plan)
            run_step3b.write_json(source / "study_status.json", status)
            with self.assertRaisesRegex(ValueError, "completed step-4a"):
                run_step4b.run(source, Path(d) / "output")
            self.assertFalse((Path(d) / "output").exists())


if __name__ == "__main__":
    unittest.main()
