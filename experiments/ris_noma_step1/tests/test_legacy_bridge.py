"""Units, common-channel bridge and inherited-moment regression checks."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.io import savemat, loadmat
from scipy.special import gammaincc
from ris_noma.config import load_config, validate_config, dbm_to_w
from ris_noma.geometry import sample_layout
from ris_noma.channels import build_bank, public_order
from ris_noma.dep import fit_willie
from ris_noma.evaluator import evaluate_with_dep
from ris_noma.objective import CachedObjective, decode_k2
from ris_noma.legacy_comparison import compare_reference
from ris_noma.pso import PSOSettings, Outcome, optimize, load_pso_settings

ROOT = Path(__file__).resolve().parents[1]


def python_fixture(c):
    """Independent literal mW formulas from the old code, NOT an Octave run.

    MAT serialization checks the bridge interface. Actual cross-runtime evidence
    still requires the exporter to run in GNU Octave.
    """
    layout = sample_layout(c, np.random.default_rng(0))
    bank = build_bank(c, layout, np.random.default_rng(11))
    x, y = np.abs(bank.h_tilde) ** 2, np.abs(bank.g_scalar_tilde) ** 2
    candidates = np.array([[.1, .5], [.2, .3], [.05, .95], [.49, .05]])
    expected = []
    # Intentionally keep literal original units and order of operations here.
    B = 10 ** (40 / 10) - 32 * (10 ** (-10 / 10) + 10 ** (-5 / 10))
    nr = 10 ** (-70 / 10)
    for mb, s in candidates:
        pa = (1 - s) * B
        rho2 = s * B / (32 * (pa / layout.loss_ar + nr))
        xx, yy = rho2 * x, rho2 * y
        rb = np.log2(1 + mb * pa * xx[:, 0] / (layout.loss_ar * layout.loss_ru[0]) /
                     (.1 ** 2 * pa * xx[:, 0] / (layout.loss_ar * layout.loss_ru[0]) + nr * yy[:, 0] / layout.loss_ru[0] + nr)).mean()
        rc = np.log2(1 + (1 - mb) * pa * xx[:, 1] / (layout.loss_ar * layout.loss_ru[1]) /
                     ((.1 ** 2 + mb) * pa * xx[:, 1] / (layout.loss_ar * layout.loss_ru[1]) + nr * yy[:, 1] / layout.loss_ru[1] + nr)).mean()
        laws = []
        for p in [(1 - mb) * pa, pa]:
            a, b = 1.01 * p / (layout.loss_ar * layout.loss_ru[-1]), nr / layout.loss_ru[-1]
            mu = a * xx[:, -1].mean() + b * yy[:, -1].mean()
            var = a ** 2 * xx[:, -1].var() + b ** 2 * yy[:, -1].var()
            laws.append((mu ** 2 / var, var / mu, mu, var))
        upper = max(mu + 8 * np.sqrt(var) for _, _, mu, var in laws)
        grid = np.linspace(0, upper, 300)
        (a0, t0, _, _), (a1, t1, _, _) = laws
        dep = gammaincc(a0, grid / t0) + 1 - gammaincc(a1, grid / t1)
        j = np.argmin(dep)
        expected.append([rho2, rb, rc, dep[j], nr + grid[j]])
    return {"schema_version": 1, "har": bank.h_ar, "hrb": bank.h_ru[:, 0],
            "hrc": bank.h_ru[:, 1], "hrw": bank.h_ru[:, 2], "phase": bank.phase,
            "X": x, "Y": y, "candidates": candidates, "expected": np.array(expected)}


class TestLegacyBridge(unittest.TestCase):
    def setUp(self):
        self.c = load_config(ROOT / "configs/legacy_physics_k1_v4.json")
        self.c["n_fading"] = 200

    def test_inherited_units_and_pathloss(self):
        c = self.c
        layout = sample_layout(c, np.random.default_rng(1))
        np.testing.assert_array_equal(layout.public_xy, [[112, 2]])
        np.testing.assert_allclose(layout.loss_ru, [572.3095724422657, 700.4041387245182, 89.91155421978291])
        self.assertAlmostEqual(layout.loss_ar, 22506.78369627292)
        self.assertAlmostEqual(float(dbm_to_w(c["total_dbm"])), 10)
        self.assertAlmostEqual(float(dbm_to_w(c["noise_dbm"]["ris"])), 1e-10)
        self.assertAlmostEqual(10 - 32 * sum(c["circuit_w"].values()), 9.98668071148746)

    def test_known_variance_divisor_and_omitted_covariance(self):
        x, y = np.array([1., 2., 4.]), np.array([3., 2., 7.])
        a = fit_willie(x, y, 2., .2, .1, 3., 4., .5, ddof=0)[-1]
        b = fit_willie(x, y, 2., .2, .1, 3., 4., .5, ddof=1)[-1]
        self.assertAlmostEqual(a["X_variance"], 14 / 9)
        self.assertAlmostEqual(b["X_variance"] / a["X_variance"], 1.5)
        for law in ["H0", "H1"]:
            self.assertAlmostEqual(a[law]["variance_paired_samples_w2"], a[law]["variance_baseline_w2"] + a[law]["omitted_covariance_term_w2"])
        with self.assertRaises(ValueError):
            validate_config(dict(self.c, moment_ddof=2))

    def test_common_channel_mat_roundtrip_and_literal_mw_formulas(self):
        fixture = python_fixture(self.c)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "synthetic_python_fixture.mat"
            savemat(path, fixture)
            report = compare_reference(self.c, loadmat(path))
        self.assertTrue(report["all_checks_pass"])
        for row in report["candidates"]:
            changes = row["v4_method_differences"]
            self.assertGreaterEqual(changes["grid_minus_stationary_dep"], -1e-12)
            self.assertGreaterEqual(changes["public_rate_reduction_from_carol_only"], -1e-12)

    def test_corrupted_channel_gain_export_is_detected(self):
        fixture = python_fixture(self.c)
        fixture["X"][0, 0] *= 2
        report = compare_reference(self.c, fixture)
        self.assertFalse(report["all_checks_pass"])
        self.assertFalse(report["channel_checks"]["X"]["pass"])

    def test_cached_k2_objective_uses_inherited_ddof_zero(self):
        c = load_config(ROOT / "configs/legacy_physics_k2_pilot.json")
        c["n_fading"] = 250
        layout = sample_layout(c, np.random.default_rng(7))
        bank = build_bank(c, layout, np.random.default_rng(8))
        order = public_order(bank)
        objective = CachedObjective(c, layout, bank, order)
        for z in [[.2, .3, .4], [.5, .5, .5], [0., 1., 1.]]:
            mb, mp, s = decode_k2(z, c["bounds"])
            summary, _, detail = evaluate_with_dep(c, layout, bank, order, mb, mp, s, .9)
            fast = objective.evaluate(z, .9)
            self.assertAlmostEqual(fast.metrics["dep_min"], summary["dep_min"], places=11)
            self.assertEqual(detail["moments"]["sample_variance_ddof"], 0)
            self.assertEqual(fast.feasible, summary["constraints"]["feasible_under_gamma_sample_model"])

    def test_fixed_los_is_shared_by_banks_and_held_across_fading(self):
        # KR -> infinity isolates deterministic LoS and phase alignment.
        c = deepcopy(self.c)
        c["rician_k"] = 1e32
        layout = sample_layout(c, np.random.default_rng(0))
        a = build_bank(c, layout, np.random.default_rng(1))
        b = build_bank(c, layout, np.random.default_rng(2))
        np.testing.assert_allclose(a.h_ar, b.h_ar, atol=1e-15)
        np.testing.assert_allclose(a.h_ar[0], np.exp(1j * np.array(c["los"]["ar_phase_rad"])), atol=1e-15)
        self.assertGreater(np.std(np.angle(a.h_ar[0])), .5)

    def test_inherited_velocity_affects_second_sweep_and_preserves_box(self):
        settings, _ = load_pso_settings(ROOT / "configs/pso_inherited_coefficients.json")
        self.assertEqual(settings.initial_velocity_scale, .1)
        self.assertEqual(settings.velocity_max, .25)
        seen = []
        def objective(z):
            seen.append(z.copy())
            return Outcome(0., 0., True)
        # No attraction: second sweep should equal clip(X + .9 V).
        settings = PSOSettings(swarm_size=3, sweeps=2, cognitive=0, social=0, initial_velocity_scale=.1, velocity_max=.25)
        optimize(objective, 2, settings, np.random.default_rng(14))
        rng = np.random.default_rng(14)
        x = rng.uniform(0, 1, (3, 2))
        v = rng.uniform(-.1, .1, (3, 2))
        np.testing.assert_array_equal(np.array(seen[:3]), x)
        np.testing.assert_allclose(seen[3:], np.clip(x + .9 * v, 0, 1))


if __name__ == "__main__":
    unittest.main()
