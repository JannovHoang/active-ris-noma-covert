"""Study error budgets, failure accounting, resume and result protection."""
from copy import deepcopy
from pathlib import Path
import contextlib
import io
import json
import tempfile
import unittest

from ris_noma.config import load_config
from ris_noma.pso import PSOSettings
import run_step3b as step3b
import run_step4a as study_runner

ROOT = Path(__file__).resolve().parents[1]


class TestStudy(unittest.TestCase):
    def setUp(self):
        self.c = load_config(ROOT / "configs/legacy_physics_k2_pilot.json")
        self.c["requirements"]["xi"] = [.9]
        self.settings = PSOSettings(swarm_size=4, sweeps=2)
        self.policy = step3b.load_policy(ROOT / "configs/validation_pilot.json")
        self.study = {"layout_ids": [1, 2, 3], "seeds": [1, 2], "study_alpha": .05}

    def fake_run(self, reject=(), review=(), calls=None):
        def runner(c, settings, policy, xis, output):
            folder = Path(output)
            folder.mkdir(parents=True, exist_ok=True)
            lid = c["layout_id"]
            if calls is not None:
                calls.append((lid, deepcopy(policy)))
            step3b.write_json(folder / "layout.json", {"public_xy_original_order": [[112, lid], [115, lid]],
                                                      "sic_order_original_ids": [1, 0], "bob_mean_gain_rule_on_label": lid not in reject})
            if lid in reject:
                step3b.write_json(folder / "status.json", {"status": "layout_rejected_by_label_bob_mean_gain_rule", "heldout_generated": False})
                return []
            step3b.write_json(folder / "selection_frozen.json", {"layout_id": lid, "selected_using_heldout": False})
            row = {"xi": xis[0], "status": "selected_on_validation_audit_review" if lid in review else "selected_on_validation_audit_pass",
                   "proposal_id": "p0001", "seed": 1, "margin": .02, "mb": .1, "m1": .4, "m2": .5, "s": .5,
                   "bob_rate_heldout": float(lid), "dep_empirical_heldout": .93, "dep_lower_heldout": .89 if lid in review else .91,
                   "heldout_point_checks_pass": True, "heldout_bound_checks_pass": lid not in review, "heldout_used_for_selection": False}
            step3b.write_json(folder / "selected_by_xi.json", [row])
            step3b.write_csv(folder / "selected_by_xi.csv", [row])
            step3b.write_json(folder / "status.json", {"status": "completed", "selected_using_heldout": False,
                "candidate_evaluations": 80, "selection_sha256_before_and_after_audit": study_runner.sha(folder / "selection_frozen.json")})
            return [row]
        return runner

    def test_alpha_is_allocated_over_all_requested_layouts_before_running(self):
        plan = study_runner.make_plan(self.c, self.settings, self.policy, self.study)
        self.assertAlmostEqual(plan["per_layout_alpha"] * 3, .05)
        self.assertEqual(plan["policy_per_layout"]["seeds"], [1, 2])
        self.assertEqual(plan["planned_candidate_evaluations_per_eligible_layout"], 80)
        self.assertEqual(plan["planned_candidate_evaluations_if_all_layouts_eligible"], 240)
        self.assertEqual(self.policy["alpha"], .05)

    def test_summary_keeps_rejections_and_audit_failures_and_mean_uses_all_selected(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            status, summary = study_runner.run_study(self.c, self.settings, self.policy, self.study, d,
                                                    run_layout=self.fake_run(reject=[2], review=[3]))
            r = summary[0]
            self.assertTrue(status["study_complete"])
            self.assertEqual(r["planned_layouts"], 3)
            self.assertEqual(r["rejected_layouts"], 1)
            self.assertEqual(r["validation_selected_count"], 2)
            self.assertEqual(r["heldout_bound_pass_count"], 1)
            self.assertEqual(r["heldout_review_count"], 1)
            self.assertEqual(r["bob_rate_heldout_mean_all_selected"], 2.)
            self.assertEqual(r["bound_pass_fraction_all_requested_layouts"], 1 / 3)

    def test_resume_skips_verified_completed_jobs(self):
        calls = []
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            study_runner.run_study(self.c, self.settings, self.policy, self.study, d, run_layout=self.fake_run(calls=calls))
            study_runner.run_study(self.c, self.settings, self.policy, self.study, d, resume=True, run_layout=self.fake_run(calls=calls))
            self.assertEqual([r[0] for r in calls], [1, 2, 3])
            for _, policy in calls:
                self.assertAlmostEqual(policy["alpha"], .05 / 3)

    def test_resume_archives_partial_job_and_preserves_completed_jobs(self):
        good = self.fake_run()
        calls = []
        def interrupted(c, *args):
            calls.append(c["layout_id"])
            if c["layout_id"] == 2:
                p = Path(args[-1]);p.mkdir(parents=True, exist_ok=True)
                (p / "partial.txt").write_text("keep interrupted output")
                raise KeyboardInterrupt()
            return good(c, *args)
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                study_runner.run_study(self.c, self.settings, self.policy, self.study, d, run_layout=interrupted)
            status = study_runner.read_json(Path(d) / "study_status.json")
            self.assertFalse(status["study_complete"])
            self.assertEqual(status["finalized_layouts"], 1)
            resumed_calls = []
            study_runner.run_study(self.c, self.settings, self.policy, self.study, d, resume=True, run_layout=self.fake_run(calls=resumed_calls))
            self.assertEqual([r[0] for r in resumed_calls], [2, 3])
            self.assertEqual((Path(d) / "layout_000002_partial_001/partial.txt").read_text(), "keep interrupted output")

    def test_resume_rejects_changed_study_plan(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            study_runner.run_study(self.c, self.settings, self.policy, self.study, d, run_layout=self.fake_run())
            changed = dict(self.study, seeds=[99])
            with self.assertRaisesRegex(ValueError, "plan/configuration/code changed"):
                study_runner.run_study(self.c, self.settings, self.policy, changed, d, resume=True, run_layout=self.fake_run())

    def test_completed_output_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            study_runner.run_study(self.c, self.settings, self.policy, self.study, d, run_layout=self.fake_run())
            path = Path(d) / "layout_000001/selected_by_xi.json"
            path.write_text("[]")
            with self.assertRaisesRegex(ValueError, "Completed result changed"):
                study_runner.run_study(self.c, self.settings, self.policy, self.study, d, resume=True, run_layout=self.fake_run())

    def test_nonempty_output_without_resume_is_protected(self):
        with tempfile.TemporaryDirectory() as d:
            marker = Path(d) / "existing.txt";marker.write_text("keep")
            with self.assertRaisesRegex(ValueError, "Nonempty study output"):
                study_runner.run_study(self.c, self.settings, self.policy, self.study, d, run_layout=self.fake_run())
            self.assertEqual(marker.read_text(), "keep")

    def test_duplicate_layouts_and_invalid_alpha_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "study.json"
            for bad in [dict(self.study, layout_ids=[1, 1]), dict(self.study, study_alpha=0), dict(self.study, seeds=[True])]:
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    study_runner.load_study(path)


if __name__ == "__main__":
    unittest.main()
