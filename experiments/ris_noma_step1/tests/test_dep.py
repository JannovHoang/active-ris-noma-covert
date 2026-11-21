"""Kiểm chứng DEP bằng trường hợp có đáp án, hai nhánh, scaling và CDF mẫu."""
from pathlib import Path
import math
import unittest
import numpy as np
from scipy.special import gammaln

from ris_noma.config import load_config
from ris_noma.channels import build_bank, public_order
from ris_noma.geometry import sample_layout
from ris_noma.random_streams import make_rng
from ris_noma.dep import (GammaLaw, fit_gamma, fit_willie, gamma_dep,
                          stationary_thresholds, minimize_gamma_dep, gamma_grid,
                          empirical_dep, empirical_at_threshold, minimize_empirical_dep,
                          lambertw_threshold_check, gamma_cdf_error)
from ris_noma.evaluator import evaluate_with_dep, constraint_report

ROOT = Path(__file__).resolve().parents[1]


class TestDEP(unittest.TestCase):
    def test_moment_matching_preserves_two_moments(self):
        law = fit_gamma(6, 12)
        self.assertAlmostEqual(law.shape, 3)
        self.assertAlmostEqual(law.scale_w, 2)
        self.assertAlmostEqual(law.mean_w, 6)
        self.assertAlmostEqual(law.shape * law.scale_w ** 2, 12)

    def test_invalid_moments_are_not_silently_repaired(self):
        for args in [(0, 1), (1, 0), (-1, 2), (1, np.nan)]:
            with self.assertRaises(ValueError):
                fit_gamma(*args)
        with self.assertRaises(ValueError):
            GammaLaw(0, 1)

    def test_exponential_pair_has_closed_form_minimum(self):
        # H0~Exp(mean=1), H1~Exp(mean=2): x*=2ln2, DEP*=3/4.
        result = minimize_gamma_dep(GammaLaw(1, 1), GammaLaw(1, 2))
        self.assertAlmostEqual(result["shift_star_w"], 2 * math.log(2), places=12)
        self.assertAlmostEqual(result["dep_min"], .75, places=12)
        self.assertAlmostEqual(result["equal_prior_error_probability"], .375, places=12)

    def test_identical_gamma_has_dep_one(self):
        law = GammaLaw(2.7, 1e-12)
        np.testing.assert_allclose(gamma_dep(np.r_[-1, 0, np.geomspace(1e-15, 1e-8, 40), np.inf], law, law), 1)
        result = minimize_gamma_dep(law, law)
        self.assertEqual(result["dep_min"], 1)
        self.assertEqual(result["stationary_shifts_w"], [])

    def test_equal_scale_degenerate_root(self):
        scale = .3
        expected = scale * math.exp((gammaln(5) - gammaln(2)) / 3)
        result = stationary_thresholds(GammaLaw(2, scale), GammaLaw(5, scale))
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], expected, places=12)

    def test_equal_shape_degenerate_root(self):
        k, t0, t1 = 3.2, .4, 1.7
        expected = k * math.log(t1 / t0) / (1 / t0 - 1 / t1)
        result = stationary_thresholds(GammaLaw(k, t0), GammaLaw(k, t1))
        self.assertAlmostEqual(result[0], expected, places=12)

    def test_two_pdf_crossings_and_both_lambert_branches(self):
        h0, h1 = GammaLaw(2, 1), GammaLaw(5, .5)
        result = minimize_gamma_dep(h0, h1)
        check = lambertw_threshold_check(h0, h1)
        self.assertEqual(len(result["stationary_shifts_w"]), 2)
        self.assertEqual(check["status"], "computed")
        np.testing.assert_allclose(result["stationary_shifts_w"], check["stationary_shifts_w"], rtol=1e-10)
        self.assertGreater(result["stationary_dep"][1], 1)  # Không clip lỗi CDF-order của fit.
        self.assertEqual(result["dep_min"], min([1, *result["stationary_dep"]]))

    def test_power_units_rescale_threshold_not_dep(self):
        base = minimize_gamma_dep(GammaLaw(2, 1), GammaLaw(5, .5))
        for factor in [1e-18, 1e-9, 1e9, 1e18]:
            result = minimize_gamma_dep(GammaLaw(2, factor), GammaLaw(5, .5 * factor))
            self.assertAlmostEqual(result["dep_min"], base["dep_min"], places=12)
            self.assertAlmostEqual(result["shift_star_w"] / factor, base["shift_star_w"], places=10)

    def test_wide_scale_ratio_and_reversed_order_use_boundary(self):
        result = minimize_gamma_dep(GammaLaw(1, 1), GammaLaw(1, 1e-20))
        self.assertEqual(result["dep_min"], 1)
        self.assertEqual(result["shift_star_w"], 0)

    def test_nearly_identical_gamma_stays_numerically_stable(self):
        result = minimize_gamma_dep(GammaLaw(5, 1), GammaLaw(5 + 1e-8, 1 + 1e-8))
        self.assertTrue(.999999 < result["dep_min"] <= 1)

    def test_grid_cannot_beat_all_stationary_candidates(self):
        rng = np.random.default_rng(20261102)
        for _ in range(24):
            shape = np.exp(rng.uniform(-3, 5, 2))
            scale = np.exp(rng.uniform(-12, 3, 2))
            h0, h1 = GammaLaw(shape[0], scale[0]), GammaLaw(shape[1], scale[1])
            result = minimize_gamma_dep(h0, h1)
            _, _, grid = gamma_grid(h0, h1, 5001)
            self.assertLessEqual(result["dep_min"], grid["dep_min"] + 1e-10)
            self.assertTrue(0 <= result["dep_min"] <= 1)

    def test_empirical_equal_samples_with_ties_have_dep_one(self):
        z = np.array([1., 1., 2., 3.])
        np.testing.assert_allclose(empirical_dep([0, 1, 1.5, 2, 3, np.inf], z, z), 1)
        self.assertEqual(minimize_empirical_dep(z, z)["dep"], 1)

    def test_empirical_separated_powers_have_zero_minimum(self):
        z0, z1 = np.array([1., 1., 2., 2.]), np.array([3., 3., 4., 4.])
        result = minimize_empirical_dep(z0, z1)
        self.assertEqual(result["dep"], 0)
        self.assertTrue(2 <= result["shift_w"] < 3)

    def test_empirical_sweep_matches_direct_detector_decisions(self):
        z0 = np.array([.1, .3, .3, .8, 1.])
        z1 = np.array([.2, .3, .6, 1., 2.])
        thresholds = np.unique(np.r_[0, z0, z1])
        direct = np.array([(np.mean(z0 > x) + np.mean(z1 <= x)) for x in thresholds])
        np.testing.assert_allclose(empirical_dep(thresholds, z0, z1), direct)
        self.assertAlmostEqual(minimize_empirical_dep(z0, z1)["dep"], direct.min())

    def test_covariance_is_diagnostic_not_added_to_baseline(self):
        x = np.array([1., 2., 3., 4., 5.])
        y = 2 * x
        _, _, z0, z1, info = fit_willie(x, y, 4, .1, .1, 3, 2, .2)
        for key, z in [("H0", z0), ("H1", z1)]:
            values = info[key]
            self.assertGreater(values["omitted_covariance_term_w2"], 0)
            self.assertAlmostEqual(values["variance_baseline_w2"] + values["omitted_covariance_term_w2"], z.var(ddof=1))
            self.assertAlmostEqual(values["shape"] * values["scale_w"] ** 2, values["variance_baseline_w2"])

    def test_no_covert_stream_gives_identical_hypotheses(self):
        x, y = np.arange(1., 10.), np.arange(3., 12.)
        h0, h1, z0, z1, _ = fit_willie(x, y, 4, 0, .1, 3, 2, .2)
        self.assertEqual(minimize_gamma_dep(h0, h1)["dep_min"], 1)
        np.testing.assert_array_equal(z0, z1)
        self.assertEqual(minimize_empirical_dep(z0, z1)["dep"], 1)

    def test_cdf_error_identifies_bad_gamma_fit(self):
        rng = np.random.default_rng(42)
        samples = rng.exponential(1, 10000)
        self.assertLess(gamma_cdf_error(samples, GammaLaw(1, 1)), .03)
        self.assertGreater(gamma_cdf_error(samples, GammaLaw(5, 1)), .5)

    def test_fixed_threshold_pointwise_error_uses_paired_samples(self):
        z0, z1 = np.array([1., 2., 3., 4.]), np.array([2., 3., 4., 5.])
        result = empirical_at_threshold(3., z0, z1)
        indicator = np.array([1., 1., 0., 1.])
        self.assertAlmostEqual(result["dep"], indicator.mean())
        self.assertAlmostEqual(result["paired_standard_error"], indicator.std(ddof=1) / 2)


class TestIntegratedEvaluator(unittest.TestCase):
    def test_k1_k2_integration_and_constraint_flags(self):
        for K in [1, 2]:
            c = load_config(ROOT / f"configs/demo_k{K}.json")
            c["n_fading"] = 400
            layout = sample_layout(c, make_rng(c["root_seed"], 0, "geometry"))
            bank = build_bank(c, layout, make_rng(c["root_seed"], 0, "label"))
            a = c["reference_allocation"]
            summary, samples, detail = evaluate_with_dep(c, layout, bank, public_order(bank), a["mb"], a["public"], a["s"], .75)
            self.assertIsInstance(summary["dep_min"], float)
            self.assertTrue(0 <= summary["dep_min"] <= 1)
            self.assertGreater(summary["lambda_star_w"], 0)
            self.assertEqual(summary["dep_model"], "baseline_without_covariance_eq_10_5")
            # Kiểm chứng phép trừ hằng noise từ V giống Z được dựng bằng X,Y.
            sw = 10 ** ((c["noise_dbm"]["willie"] - 30) / 10)
            np.testing.assert_allclose(samples["v0_willie"] - sw, samples["z0_willie"], rtol=1e-12)
            np.testing.assert_allclose(samples["v1_willie"] - sw, samples["z1_willie"], rtol=1e-12)
            low = constraint_report(summary, .5)
            high = constraint_report(summary, .999999)
            self.assertTrue(low["checks"]["covertness_gamma"])
            self.assertFalse(high["checks"]["covertness_gamma"])


if __name__ == "__main__":
    unittest.main()
