"""Regression and variable-size tests for the 7a foundation, not ML accuracy."""
from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np
import torch

from ris_noma.channels import build_bank, public_order
from ris_noma.config import load_config
from ris_noma.evaluator import evaluate_with_dep
from ris_noma.geometry import sample_layout
from ris_noma.graph_data import statistical_graph as old_graph
from ris_noma.learning_models import AllocationGNN
from ris_noma.multik_core import (MultiKObjective, assess_allocation, decode_allocation,
                                 extend_config, original_allocation, physical_allocation)
from ris_noma.multik_graph import fit_normalizer, pack_graphs, statistical_graph, unpack_predictions, validate_graph
from ris_noma.multik_models import VariableAllocationGNN, balanced_allocation_loss, tensor_batch
from ris_noma.objective import CachedObjective, decode_k2
from ris_noma.pso import PSOSettings, optimize

ROOT = Path(__file__).resolve().parents[1]


def context(k, samples=256):
    base = load_config(ROOT / "configs/legacy_physics_k2_pilot.json")
    c = extend_config(base, k)
    c["n_fading"] = samples
    layout = sample_layout(c, np.random.default_rng(13))
    bank = build_bank(c, layout, np.random.default_rng(17))
    return c, layout, bank, public_order(bank)


class MultiKCoreTests(unittest.TestCase):
    def test_decoder_bounds_and_roundtrip(self):
        rng = np.random.default_rng(7)
        for k in (1, 2, 3, 5, 8):
            c, *_ = context(k, 32)
            order = rng.permutation(k)
            for z in rng.random((20, k + 1)):
                mb, mp, s = decode_allocation(z, k, c["bounds"])
                y, inverse = original_allocation(mb, mp, s, order, c["bounds"])
                self.assertAlmostEqual(sum(y[:-1]), 1.)
                self.assertTrue(np.all(y[:-1] >= 0))
                actual = decode_allocation(inverse, k, c["bounds"])
                np.testing.assert_allclose(np.r_[actual[0], actual[1], actual[2]], np.r_[mb, mp, s], atol=1e-12)

    def test_decoder_k2_matches_old(self):
        c, *_ = context(2, 32)
        for z in np.random.default_rng(9).random((30, 3)):
            a, b = decode_allocation(z, 2, c["bounds"]), decode_k2(z, c["bounds"])
            np.testing.assert_array_equal(np.r_[a[0], a[1], a[2]], np.r_[b[0], b[1], b[2]])

    def test_simplex_cut_distribution_is_uniform(self):
        c, *_ = context(3, 32)
        rng = np.random.default_rng(91)
        cuts = np.sort(rng.random((10000, 2)), axis=1)
        fractions = np.diff(np.c_[np.zeros(len(cuts)), cuts, np.ones(len(cuts))], axis=1)
        np.testing.assert_allclose(fractions.mean(0), np.ones(3) / 3, atol=.008)

    def test_invalid_allocation_rejected(self):
        c, *_ = context(3, 32)
        for z in ([.1, .2], [0., .2, .3, 1.1], [np.nan, .2, .3, .5]):
            with self.assertRaises(ValueError):
                decode_allocation(z, 3, c["bounds"])
        with self.assertRaises(ValueError):
            physical_allocation([.2, .2, .2, .1, .5], 3, c["bounds"])
        with self.assertRaises(ValueError):
            decode_allocation([.1], 0, c["bounds"])

    def test_explicit_los_preserves_existing_links(self):
        base = load_config(ROOT / "configs/legacy_physics_k2_pilot.json")
        a, b = extend_config(base, 3), extend_config(base, 5)
        np.testing.assert_array_equal(a["los"]["ru_phase_rad"][:3], base["los"]["ru_phase_rad"][:3])
        np.testing.assert_array_equal(a["los"]["ru_phase_rad"][-1], base["los"]["ru_phase_rad"][-1])
        np.testing.assert_array_equal(np.asarray(a["los"]["ru_phase_rad"])[1:-1], np.asarray(b["los"]["ru_phase_rad"])[1:4])
        self.assertEqual(extend_config(base, 2), base)
        self.assertEqual(base["n_public"], 2)

    def test_heterogeneous_extensions_need_explicit_values(self):
        base = load_config(ROOT / "configs/legacy_physics_k2_pilot.json")
        base["requirements"]["qos_public"] = [.2, .3]
        with self.assertRaises(ValueError):
            extend_config(base, 3)
        c = extend_config(base, 3, {"qos_public": [.2, .3, .4]})
        self.assertEqual(c["requirements"]["qos_public"], [.2, .3, .4])

    def test_cached_objective_matches_full_evaluator_all_k(self):
        rng = np.random.default_rng(42)
        for k in (1, 2, 3, 4, 5):
            c, layout, bank, order = context(k)
            fast = MultiKObjective(c, layout, bank, order)
            for z in rng.random((8, k + 1)):
                result = fast.evaluate(z, .85)
                mb, mp, s = decode_allocation(z, k, c["bounds"])
                summary, _, _ = evaluate_with_dep(c, layout, bank, order, mb, mp, s, .85)
                np.testing.assert_allclose(result.score, summary["bob_mean_rate_bit_s_hz"], atol=1e-12)
                np.testing.assert_allclose(result.metrics["public_mean_rate_bit_s_hz"], summary["public_mean_rate_bit_s_hz"], atol=1e-12)
                np.testing.assert_allclose(result.metrics["dep_min"], summary["dep_min"], atol=1e-12)
                self.assertEqual(result.metrics["bob_outage_estimate"], summary["bob_outage_estimate"])
                self.assertEqual(result.feasible, summary["constraints"]["feasible_under_gamma_sample_model"])

    def test_cached_k2_matches_old_objective(self):
        c, layout, bank, order = context(2)
        new, old = MultiKObjective(c, layout, bank, order), CachedObjective(c, layout, bank, order)
        for z in np.random.default_rng(5).random((20, 3)):
            a, b = new.evaluate(z, .9), old.evaluate(z, .9)
            np.testing.assert_allclose([a.score, a.violation, a.metrics["dep_min"]], [b.score, b.violation, b.metrics["dep_min"]], atol=1e-12)
            self.assertEqual(a.feasible, b.feasible)

    def test_sic_and_qos_failures_are_rejected(self):
        for k in (1, 3, 5):
            c, layout, bank, order = context(k)
            c["requirements"]["target_public"] = [10.] * k
            result = MultiKObjective(c, layout, bank, order).evaluate(np.full(k + 1, .5), .75)
            self.assertFalse(result.feasible)
            self.assertEqual(result.metrics["bob_outage_estimate"], 1.)
            mb, mp, s = decode_allocation(np.linspace(.1, .9, k + 1), k, c["bounds"])
            y, _ = original_allocation(mb, mp, s, order, c["bounds"])
            _, screen = assess_allocation(c, layout, bank, order, y, .75, 1, .05)
            self.assertFalse(screen["accepted"])
            self.assertFalse(screen["checks"]["positive_sic_ceiling_margins"])

    def test_pso_uses_k_plus_one_coordinates(self):
        for k in (1, 2, 3, 5):
            c, layout, bank, order = context(k, 64)
            objective = MultiKObjective(c, layout, bank, order)
            result = optimize(lambda z: objective.evaluate(z, .8), k + 1,
                              PSOSettings(swarm_size=6, sweeps=2), np.random.default_rng(23))
            self.assertEqual(result.position.shape, (k + 1,))
            self.assertEqual(result.evaluations, 12)


class MultiKGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        cls.graphs, cls.config = [], None
        for k in (1, 2, 3, 4, 5):
            c, layout, bank, order = context(k)
            cls.graphs.append(statistical_graph(c, layout, bank, order, .85))
            cls.config = c
        cls.normalizer = fit_normalizer(cls.graphs[1:4])

    def model(self):
        return VariableAllocationGNN(self.config["bounds"], hidden=16, layers=2).eval()

    def test_k2_graph_exactly_matches_existing_schema_values(self):
        c, layout, bank, order = context(2)
        old = old_graph(c, layout, bank, order, .85)
        for key in ("node_raw", "global_raw", "edge_index", "edge_attr"):
            np.testing.assert_array_equal(old[key], self.graphs[1][key])

    def test_graph_shape_and_public_count_validation(self):
        for g in self.graphs:
            k = g["n_public"]
            self.assertEqual(validate_graph(g), k)
            self.assertEqual(np.asarray(g["node_raw"]).shape, (k + 1, 15))
            self.assertEqual(np.asarray(g["edge_index"]).shape, (2, k * (k + 1)))
        bad = deepcopy(self.graphs[2]); bad["global_raw"][15] = 2
        with self.assertRaises(ValueError):
            validate_graph(bad)

    def test_one_model_mixed_k_and_power_simplex(self):
        model = self.model()
        before = sum(p.numel() for p in model.parameters())
        data = tensor_batch(pack_graphs(self.graphs, self.normalizer))
        with torch.inference_mode():
            y = model(**data)
        self.assertEqual(before, sum(p.numel() for p in model.parameters()))
        np.testing.assert_allclose(y[:, :-1].sum(-1), 1., atol=2e-7)
        self.assertTrue(bool(torch.isfinite(y).all()))
        self.assertTrue(bool((y[:, 1:-1][~data["node_mask"][:, 1:]] == 0).all()))
        result = unpack_predictions(y.numpy(), self.graphs)
        self.assertEqual([len(a) for a in result], [g["n_public"] + 2 for g in self.graphs])

    def test_padding_does_not_change_predictions(self):
        model = self.model()
        with torch.inference_mode():
            mixed = model(**tensor_batch(pack_graphs(self.graphs, self.normalizer))).numpy()
            for i, g in enumerate(self.graphs):
                single = model(**tensor_batch(pack_graphs([g], self.normalizer))).numpy()
                np.testing.assert_allclose(unpack_predictions(mixed[i:i+1], [g])[0], single[0], atol=1e-7)

    def test_padded_node_and_edge_values_are_ignored(self):
        model = self.model()
        data = tensor_batch(pack_graphs(self.graphs, self.normalizer))
        with torch.inference_mode():
            y = model(**data)
            data["node"][~data["node_mask"]] = 35.
            data["edge_attr"][~data["edge_mask"]] = -12.
            z = model(**data)
        torch.testing.assert_close(y, z, atol=1e-7, rtol=1e-6)

    def test_public_permutation_equivariance(self):
        model = self.model()
        for g in self.graphs:
            n = g["n_public"] + 1
            permutation = np.r_[0, np.arange(n - 1, 0, -1)]
            inverse = np.argsort(permutation)
            changed = deepcopy(g)
            changed["node_raw"] = np.asarray(g["node_raw"])[permutation].tolist()
            changed["edge_index"] = inverse[np.asarray(g["edge_index"])].tolist()
            with torch.inference_mode():
                a = model(**tensor_batch(pack_graphs([g], self.normalizer))).numpy()[0]
                b = model(**tensor_batch(pack_graphs([changed], self.normalizer))).numpy()[0]
            np.testing.assert_allclose(b[:-1], a[:-1][permutation], atol=1e-7)
            self.assertAlmostEqual(float(a[-1]), float(b[-1]), places=6)

    def test_legacy_weights_load_and_predict_same_for_k2(self):
        torch.manual_seed(12)
        old = AllocationGNN(self.config["bounds"], hidden=16, layers=2).eval()
        new = self.model(); new.load_state_dict(old.state_dict(), strict=True)
        data = tensor_batch(pack_graphs([self.graphs[1]], self.normalizer))
        with torch.inference_mode():
            a = old(**{k: data[k] for k in ("node", "global_features", "edge_index", "edge_attr")})
            b = new(**data)
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)

    def test_loss_has_no_padded_gradient(self):
        data = tensor_batch(pack_graphs(self.graphs, self.normalizer))
        prediction = torch.rand((len(self.graphs), data["node"].shape[1] + 1), requires_grad=True)
        target = torch.zeros_like(prediction)
        value = balanced_allocation_loss(prediction, target, data["node_mask"])
        value.backward()
        self.assertTrue(bool((prediction.grad[:, 1:-1][~data["node_mask"][:, 1:]] == 0).all()))
        self.assertTrue(bool((prediction.grad[:, 0] != 0).all()))

    def test_k2_loss_matches_previous_weighted_mse(self):
        a, b = torch.rand((5, 4)), torch.rand((5, 4))
        expected = ((a - b).square() * torch.tensor([32., 1., 1., 1.])).sum(-1).mean() / 35.
        actual = balanced_allocation_loss(a, b, torch.ones((5, 3), dtype=torch.bool))
        torch.testing.assert_close(actual, expected)

    def test_mixed_batch_backward_and_optimizer_step(self):
        model = self.model(); model.train()
        targets = [[.2] + [.8/g["n_public"]] * g["n_public"] + [.5] for g in self.graphs]
        data = tensor_batch(pack_graphs(self.graphs, self.normalizer, targets))
        target = data.pop("target")
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        before = model.bob_head[0].weight.detach().clone()
        loss = balanced_allocation_loss(model(**data), target, data["node_mask"])
        loss.backward(); optimizer.step()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertFalse(torch.equal(before, model.bob_head[0].weight))

    def test_invalid_mask_rejected(self):
        model = self.model()
        data = tensor_batch(pack_graphs([self.graphs[1]], self.normalizer))
        data["node_mask"][:, 1:] = False
        with self.assertRaises(ValueError):
            model(**data)

    def test_training_does_not_filter_labels_using_heldout(self):
        import json
        from tempfile import TemporaryDirectory
        from ris_noma.multik_demo import train_shared, write_json
        config = json.loads((ROOT / "configs/multik_foundation_7a.json").read_text())
        config["training"].update(epochs=1, hidden=16, layers=1)
        records = []
        for split in ("train", "validation", "test"):
            for g in self.graphs[1:4]:
                k = g["n_public"]
                records.append({"split": split, "n_public": k, "label_available": True,
                    "graph": g, "allocation": [.2] + [.8/k]*k + [.5],
                    "label_audit": {"passed": False}})
        with TemporaryDirectory() as temp:
            output = Path(temp)
            plan = {"config": config}
            write_json(output / "plan.json", plan)
            result = train_shared(output, plan, {"base_config": self.config}, records)
            self.assertEqual(result["labeled_counts"], {"train": 3, "validation": 3, "test": 3})

    def test_corrupted_sic_relation_rejected(self):
        g = deepcopy(self.graphs[2]); g["edge_attr"][0] = [0., 0.]
        with self.assertRaises(ValueError):
            validate_graph(g)


class MultiKProtocolTests(unittest.TestCase):
    def config(self):
        import json
        return json.loads((ROOT / "configs/multik_foundation_7a.json").read_text())

    def test_group_split_and_unseen_k_guards(self):
        from ris_noma.multik_demo import validate_settings
        c = self.config(); validate_settings(c)
        c["splits"]["test"] = c["splits"]["train"][:1]
        with self.assertRaises(ValueError):
            validate_settings(c)
        c = self.config(); c["unseen_public_counts"] = [4]
        with self.assertRaises(ValueError):
            validate_settings(c)

    def test_new_immutable_record_and_receipt_detect_changes(self):
        from tempfile import TemporaryDirectory
        from ris_noma.multik_demo import committed, immutable
        with TemporaryDirectory() as temp:
            path = Path(temp) / "row.json"
            committed(path, lambda: {"a": 1})
            self.assertEqual(committed(path, lambda: {"a": 2}), {"a": 1})
            with self.assertRaises(ValueError):
                immutable(path, {"a": 2})
            path.write_text('{"a":3}')
            with self.assertRaises(ValueError):
                committed(path, lambda: {"a": 4})


if __name__ == "__main__":
    unittest.main()
