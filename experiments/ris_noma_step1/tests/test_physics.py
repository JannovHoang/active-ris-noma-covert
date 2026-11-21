"""Kiểm tra mô hình khoa học; chạy bằng unittest có sẵn trong Python."""
from pathlib import Path
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np

from ris_noma.config import load_config, validate_config, dbm_to_w
from ris_noma.geometry import path_loss, sample_layout
from ris_noma.random_streams import make_rng
from ris_noma.channels import complex_normal, rician, build_bank, public_order
from ris_noma.metrics import evaluate_candidate

ROOT = Path(__file__).resolve().parents[1]


class TestPhysics(unittest.TestCase):
    def setUp(self):
        self.c = load_config(ROOT / "configs/demo_k2.json")
        self.c["n_fading"] = 300
        self.layout = sample_layout(self.c, make_rng(7, 0, "geometry"))
        self.bank = build_bank(self.c, self.layout, make_rng(7, 0, "label"))
        self.order = public_order(self.bank)

    def evaluate(self, mb=.1, mp=(.6, .3), s=.5):
        return evaluate_candidate(self.c, self.layout, self.bank, self.order, mb, mp, s)

    def test_units_and_pathloss(self):
        self.assertAlmostEqual(float(dbm_to_w(40)), 10)
        self.assertAlmostEqual(float(dbm_to_w(0)), .001)
        self.assertAlmostEqual(float(path_loss(10, .001, 2, 1)), 100000)

    def test_normal_and_rician_moments(self):
        rng = np.random.default_rng(20261102)
        z = complex_normal(rng, (150000,))
        self.assertLess(abs(z.real.var() - .5), .012)
        self.assertLess(abs(z.imag.var() - .5), .012)
        for K in [0, 1, 10]:
            h = rician(rng, (150000,), K, 1 + 0j)
            self.assertLess(abs(np.mean(np.abs(h) ** 2) - 1), .012)
            self.assertLess(abs(h.mean() - np.sqrt(K / (K + 1))), .012)

    def test_rayleigh_limit_has_exponential_power(self):
        h = rician(np.random.default_rng(11), (150000,), 0, 1 + 0j)
        empirical_cdf_at_1 = np.mean(np.abs(h) ** 2 < 1)
        self.assertLess(abs(empirical_cdf_at_1 - (1 - np.exp(-1))), .006)

    def test_streams_reproducible_and_separate(self):
        np.testing.assert_array_equal(make_rng(7, 2, "label").normal(size=50), make_rng(7, 2, "label").normal(size=50))
        self.assertFalse(np.array_equal(make_rng(7, 2, "label").normal(size=50), make_rng(7, 2, "evaluation").normal(size=50)))
        self.assertFalse(np.array_equal(make_rng(7, 2, "geometry").normal(size=50), make_rng(7, 3, "geometry").normal(size=50)))

    def test_geometry_fixed_and_within_square(self):
        center = np.asarray(self.c["positions"]["public_center"])
        self.assertTrue(np.all(np.abs(self.layout.public_xy - center) <= self.c["public_half_width_m"]))
        self.assertEqual(self.bank.h_ar.shape, (300, 32))
        self.assertEqual(self.bank.h_ru.shape, (300, 4, 32))
        self.assertEqual(self.layout.receiver_xy.shape, (4, 2))

    def test_phase_alignment_and_cached_power(self):
        aligned = np.sum(np.abs(self.bank.h_ar) * np.abs(self.bank.h_ru[:, 0, :]), axis=1)
        np.testing.assert_allclose(self.bank.h_tilde[:, 0], aligned, rtol=1e-12, atol=1e-12)
        rho = 3.7
        h_direct = np.sum(self.bank.h_ar[:, None, :] * self.bank.h_ru * (rho * self.bank.phase[:, None, :]), axis=2)
        gain_direct = np.abs(h_direct) ** 2 / (self.layout.loss_ar * self.layout.loss_ru)
        np.testing.assert_allclose(gain_direct, rho ** 2 * self.bank.signal_gain_tilde, rtol=1e-12)
        self.assertFalse(self.bank.h_ar.flags.writeable)

    def test_scalar_ris_noise_can_cancel_coherently(self):
        # Phân biệt |sum complex terms|² với sum |terms|².
        c = deepcopy(self.c)
        c["n_ris"], c["n_fading"] = 2, 2
        h_ar = np.ones((2, 2), complex)
        h_ru = np.ones((2, 4, 2), complex)
        h_ru[:, 0, 1] = -1  # Bob => RIS phase [1,-1].
        with patch("ris_noma.channels.rician", side_effect=[h_ar, h_ru]):
            bank = build_bank(c, self.layout, np.random.default_rng(1))
        np.testing.assert_allclose(bank.noise_gain_tilde[:, 1:], 0, atol=1e-28)
        np.testing.assert_allclose(bank.h_tilde[:, 0], 2, atol=1e-12)

    def test_k1_reduces_to_original_bob_carol_equations(self):
        c = load_config(ROOT / "configs/demo_k1.json")
        c["n_fading"] = 300
        layout = sample_layout(c, make_rng(1, 0, "geometry"))
        bank = build_bank(c, layout, make_rng(1, 0, "label"))
        summary, samples = evaluate_candidate(c, layout, bank, np.array([0]), .1, [.9], .5)
        power = summary["power"]
        pa, rho2 = power["pa_w"], power["rho2"]
        g, q = rho2 * bank.signal_gain_tilde, rho2 * bank.noise_gain_tilde
        sr = dbm_to_w(c["noise_dbm"]["ris"])
        sb = dbm_to_w(c["noise_dbm"]["bob"])
        sc = dbm_to_w(c["noise_dbm"]["public"][0])
        pb, pc, k2 = .1 * pa, .9 * pa, c["kappa"] ** 2
        gamma_b = pb * g[:, 0] / (k2 * pa * g[:, 0] + sr * q[:, 0] + sb)
        gamma_bc = pc * g[:, 0] / ((pb + k2 * pa) * g[:, 0] + sr * q[:, 0] + sb)
        gamma_c = pc * g[:, 1] / ((pb + k2 * pa) * g[:, 1] + sr * q[:, 1] + sc)
        np.testing.assert_allclose(samples["gamma_bob"], gamma_b)
        np.testing.assert_allclose(samples["gamma_public"][:, 0, 0], gamma_bc)
        np.testing.assert_allclose(samples["gamma_public"][:, 1, 0], gamma_c)
        self.assertAlmostEqual(summary["public_mean_rate_bit_s_hz"][0], np.minimum(np.log2(1 + gamma_bc), np.log2(1 + gamma_c)).mean())

    def test_k2_sinrs_direct_power_and_rate_ceiling(self):
        summary, samples = self.evaluate()
        pa = summary["power"]["pa_w"]
        rho2 = summary["power"]["rho2"]
        cols = np.r_[0, 1 + self.order, 3]
        g, q = rho2 * self.bank.signal_gain_tilde[:, cols], rho2 * self.bank.noise_gain_tilde[:, cols]
        sigma = dbm_to_w(np.r_[self.c["noise_dbm"]["bob"], np.asarray(self.c["noise_dbm"]["public"])[self.order], self.c["noise_dbm"]["willie"]])
        noise = dbm_to_w(self.c["noise_dbm"]["ris"]) * q + sigma
        direct1 = .6 * pa * g / ((.1 + .3 + .01) * pa * g + noise)
        direct2 = .3 * pa * g / ((.1 + .01) * pa * g + noise)
        np.testing.assert_allclose(samples["gamma_public"][:, :, 0], direct1)
        np.testing.assert_allclose(samples["gamma_public"][:, :, 1], direct2)
        self.assertTrue(np.all(samples["gamma_bob"] < .1 / .01))

    def test_outage_chain_equals_threshold_including_failure(self):
        # evaluate_candidate tự kiểm tra hai cách trên từng realization.
        for mp in [[.6, .3], [.0, .9], [.89, .01]]:
            summary, samples = self.evaluate(mp=mp)
            self.assertAlmostEqual(summary["bob_outage_estimate"], samples["bob_outage"].mean())
            self.assertTrue(0 <= summary["bob_outage_estimate"] <= 1)
        summary, _ = self.evaluate(mp=[0, .9])
        self.assertEqual(summary["bob_outage_estimate"], 1)

    def test_nonzero_outage_with_target_near_ceiling(self):
        # Chọn target trong vùng chuyển tiếp để kiểm tra ngưỡng có cả pass/fail.
        _, samples = self.evaluate()
        self.c["requirements"]["target_bob"] = float(np.median(samples["bob_rate"]))
        summary, _ = self.evaluate()
        self.assertTrue(.4 < summary["bob_outage_estimate"] < .6)

    def test_willie_keeps_public_power_and_ris_under_both_hypotheses(self):
        summary, samples = self.evaluate()
        pa, rho2 = summary["power"]["pa_w"], summary["power"]["rho2"]
        gw = rho2 * self.bank.signal_gain_tilde[:, -1]
        np.testing.assert_allclose(samples["v1_willie"] - samples["v0_willie"], 1.01 * .1 * pa * gw, rtol=1e-12, atol=0)
        self.assertIsNone(summary["dep_min"])
        self.assertEqual(summary["full_feasibility"], "not_evaluated_dep_missing")

    def test_public_physical_id_permutation_preserves_metrics(self):
        # Noise và target khác nhau để phát hiện lỗi gắn nhầm requirement khi sắp SIC.
        self.c["noise_dbm"]["public"] = [-50, -70]
        self.c["requirements"]["target_public"] = [.1, .4]
        self.c["requirements"]["qos_public"] = [.2, .6]
        first, _ = self.evaluate()
        cols = [0, 2, 1, 3]
        bank = replace(self.bank, h_ru=self.bank.h_ru[:, cols, :], h_tilde=self.bank.h_tilde[:, cols],
                       g_scalar_tilde=self.bank.g_scalar_tilde[:, cols],
                       signal_gain_tilde=self.bank.signal_gain_tilde[:, cols],
                       noise_gain_tilde=self.bank.noise_gain_tilde[:, cols])
        c = deepcopy(self.c)
        c["noise_dbm"]["public"].reverse()
        c["requirements"]["target_public"].reverse()
        c["requirements"]["qos_public"].reverse()
        second, _ = evaluate_candidate(c, self.layout, bank, 1 - self.order, .1, [.6, .3], .5)
        for key in ["bob_mean_rate_bit_s_hz", "public_mean_rate_bit_s_hz", "bob_outage_estimate", "public_outage_estimate", "qos_pass_per_public"]:
            np.testing.assert_array_equal(first[key], second[key])

    def test_invalid_configuration_cannot_silently_use_demo_values(self):
        with self.assertRaisesRegex(ValueError, "Cần điền tham số baseline"):
            load_config(ROOT / "configs/baseline_template.json")
        c = deepcopy(self.c)
        c["profile"] = "baseline"
        with self.assertRaisesRegex(ValueError, "LoS=1"):
            validate_config(c)
        with self.assertRaisesRegex(ValueError, "simplex"):
            self.evaluate(mp=[.6, .5])


if __name__ == "__main__":
    unittest.main()
