"""PSO: nghiệm biết trước, constraint dominance, budget và đối chiếu vật lý."""
from pathlib import Path
import unittest
import numpy as np

from ris_noma.pso import Outcome, PSOSettings, better, optimize, load_pso_settings
from ris_noma.config import load_config
from ris_noma.geometry import sample_layout
from ris_noma.channels import build_bank, public_order
from ris_noma.random_streams import make_rng
from ris_noma.objective import CachedObjective, decode_k2
from ris_noma.evaluator import evaluate_with_dep

ROOT = Path(__file__).resolve().parents[1]


class TestPSO(unittest.TestCase):
    def test_feasible_beats_arbitrarily_large_infeasible_score(self):
        feasible = Outcome(.1, 0., True)
        infeasible = Outcome(1e12, 1e-12, False)
        self.assertTrue(better(feasible, infeasible))
        self.assertFalse(better(infeasible, feasible))

    def test_infeasible_comparison_uses_violation_before_score(self):
        self.assertTrue(better(Outcome(1, .1, False), Outcome(100, .2, False)))
        self.assertTrue(better(Outcome(2, .1, False), Outcome(1, .1, False)))

    def test_known_concave_objective_is_solved(self):
        result = optimize(lambda z: Outcome(-float(np.sum((z - .3) ** 2)), 0., True), 3,
                          PSOSettings(), np.random.default_rng(7))
        self.assertGreater(result.outcome.score, -1e-6)
        self.assertEqual(result.evaluations, 1800)
        self.assertEqual(result.feasible_evaluations, 1800)
        rates = [h["gbest_score"] for h in result.history]
        self.assertTrue(np.all(np.diff(rates) >= 0))
        self.assertAlmostEqual(result.history[1]["inertia_used_for_update"], .9)
        self.assertAlmostEqual(result.history[-1]["inertia_used_for_update"], .4)

    def test_constrained_optimum_respects_boundary(self):
        def objective(z):
            v = float(max(0, z[0] - .3))
            return Outcome(float(z[0]), v, v == 0)
        result = optimize(objective, 1, PSOSettings(swarm_size=15, sweeps=50), np.random.default_rng(4), [[.25]])
        self.assertTrue(result.outcome.feasible)
        self.assertTrue(.299 < result.position[0] <= .3)

    def test_all_infeasible_is_reported_not_converted_to_feasible(self):
        result = optimize(lambda z: Outcome(float(z[0] * 1e8), float(1 + z[0]), False), 1,
                          PSOSettings(swarm_size=8, sweeps=12), np.random.default_rng(2))
        self.assertFalse(result.outcome.feasible)
        self.assertEqual(result.feasible_evaluations, 0)
        self.assertAlmostEqual(result.outcome.violation, 1)

    def test_single_sweep_is_initialization_only(self):
        result = optimize(lambda z: Outcome(float(z.sum()), 0., True), 3,
                          PSOSettings(swarm_size=7, sweeps=1), np.random.default_rng(1))
        self.assertEqual(result.evaluations, 7)
        self.assertIsNone(result.history[0]["inertia_used_for_update"])

    def test_position_projection_keeps_evaluated_points_in_box(self):
        seen = []
        def objective(z):
            self.assertTrue(np.all((z >= 0) & (z <= 1)))
            seen.append(z)
            return Outcome(float(z.sum()), 0., True)
        result = optimize(objective, 3, PSOSettings(swarm_size=8, sweeps=8, cognitive=10, social=10, velocity_max=2), np.random.default_rng(8))
        self.assertEqual(len(seen), result.evaluations)

    def test_reproducible_with_same_seed(self):
        f = lambda z: Outcome(-float(np.sum((z - .2) ** 2)), 0., True)
        settings = PSOSettings(swarm_size=9, sweeps=10)
        a = optimize(f, 3, settings, np.random.default_rng(11))
        b = optimize(f, 3, settings, np.random.default_rng(11))
        np.testing.assert_array_equal(a.position, b.position)
        self.assertEqual(a.history, b.history)

    def test_invalid_settings_and_nonfinite_fitness_are_rejected(self):
        for kwargs in [{"sweeps": 0}, {"swarm_size": -1}, {"velocity_max": 0}, {"cognitive": np.nan}]:
            with self.assertRaises(ValueError):
                PSOSettings(**kwargs)
        with self.assertRaisesRegex(ValueError, "hữu hạn"):
            optimize(lambda z: Outcome(np.nan, 0., True), 1, PSOSettings(sweeps=1), np.random.default_rng(1))
        with self.assertRaisesRegex(ValueError, "thống nhất"):
            optimize(lambda z: Outcome(1, 1., True), 1, PSOSettings(sweeps=1), np.random.default_rng(1))

    def test_pilot_settings_are_explicit_and_budget_is_1800(self):
        settings, raw = load_pso_settings(ROOT / "configs/pso_pilot.json")
        self.assertIn("pilot", raw["parameter_status"])
        self.assertEqual(settings.swarm_size * settings.sweeps, 1800)


class TestCachedObjective(unittest.TestCase):
    def setUp(self):
        self.c = load_config(ROOT / "configs/demo_k2.json")
        self.c["n_fading"] = 350
        self.layout = sample_layout(self.c, make_rng(7, 0, "geometry"))
        self.bank = build_bank(self.c, self.layout, make_rng(7, 0, "label"))
        self.order = public_order(self.bank)
        self.objective = CachedObjective(self.c, self.layout, self.bank, self.order)

    def compare_reference(self, z, xi=.9):
        fast = self.objective.evaluate(z, xi)
        mb, mp, s = decode_k2(z, self.c["bounds"])
        ref, _, _ = evaluate_with_dep(self.c, self.layout, self.bank, self.order, mb, mp, s, xi)
        for name in ["bob_mean_rate_bit_s_hz", "public_mean_rate_bit_s_hz", "dep_min", "bob_outage_estimate"]:
            np.testing.assert_allclose(fast.metrics[name], ref[name], rtol=1e-10, atol=1e-12)
        self.assertEqual(fast.feasible, ref["constraints"]["feasible_under_gamma_sample_model"])
        return fast

    def test_decoder_enforces_simplex_and_declared_bounds(self):
        for z in [[0, 0, 0], [1, 1, 1], [.2, .8, .7]]:
            mb, mp, s = decode_k2(z, self.c["bounds"])
            self.assertAlmostEqual(mb + mp.sum(), 1)
            self.assertTrue(np.all(mp >= 0))
            self.assertTrue(.05 <= mb <= .49)
            self.assertTrue(.05 <= s <= .95)
        with self.assertRaises(ValueError):
            decode_k2([0, 2, 0], self.c["bounds"])

    def test_cached_metrics_match_step2_on_random_points_and_boundaries(self):
        rng = np.random.default_rng(15)
        for z in [[0, 0, 0], [1, 1, 1], *rng.uniform(size=(20, 3))]:
            self.compare_reference(z)

    def test_heterogeneous_noise_qos_and_targets_preserve_sic_mapping(self):
        self.c["noise_dbm"]["public"] = [-50, -70]
        self.c["requirements"]["qos_public"] = [.2, .5]
        self.c["requirements"]["target_public"] = [.1, .4]
        self.c["requirements"]["target_bob"] = 4
        self.objective = CachedObjective(self.c, self.layout, self.bank, self.order)
        for z in [[.1, .6, .5], [.8, .2, .1]]:
            self.compare_reference(z)

    def test_repeat_evaluation_does_not_resample_or_mutate_bank(self):
        z = np.array([.2, .6, .5])
        before = self.bank.h_ar.copy()
        a, b = self.objective.evaluate(z, .9), self.objective.evaluate(z, .9)
        self.assertEqual(a, b)
        np.testing.assert_array_equal(before, self.bank.h_ar)

    def test_xi_changes_constraint_not_physical_metrics(self):
        z = np.array([.1, .6, .5])
        a, b = self.objective.evaluate(z, .5), self.objective.evaluate(z, .9999)
        self.assertEqual(a.score, b.score)
        self.assertEqual(a.metrics["dep_min"], b.metrics["dep_min"])
        self.assertTrue(a.feasible)
        self.assertFalse(b.feasible)

    def test_pso_integrates_with_reference_evaluator(self):
        result = optimize(lambda z: self.objective.evaluate(z, .9), 3,
                          PSOSettings(swarm_size=10, sweeps=8), np.random.default_rng(21))
        self.assertEqual(result.evaluations, 80)
        self.assertTrue(result.outcome.feasible)
        ref = self.compare_reference(result.position)
        self.assertEqual(result.outcome.score, ref.score)


if __name__ == "__main__":
    unittest.main()
