"""6f loss gradients, finite validation selection, split integrity and real checkpoint replay."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

try:
    import torch
    from ris_noma.physics_training import load_matched, select_checkpoints, validate_config
    from ris_noma.physics_training_fit import train_fit, weighted_mse
    from ris_noma.learning_data import INPUT_KEYS, TARGET_KEY, read_json, sha256
    from ris_noma.learning_train import load_checkpoint, predict
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None


def config():
    return {"test_layout_ids": [500, 501], "epochs": 2, "snapshot_epochs": [1, 2],
            "mb_error_weight": 32., "bootstrap_repetitions": 1000}


@unittest.skipIf(torch is None, "Install the step 6b PyTorch environment")
class TestPhysicsTraining(unittest.TestCase):
    def test_uniform_loss_and_gradients_match_mse(self):
        a = torch.tensor([[.2, .3, .5, .4]], requires_grad=True)
        target = torch.tensor([[.1, .4, .5, .6]])
        reference = torch.nn.functional.mse_loss(a, target)
        loss = weighted_mse(a, target, [1., 1., 1., 1.])
        self.assertEqual(float(loss.detach()), float(reference.detach()))
        np.testing.assert_array_equal(torch.autograd.grad(loss, a)[0], torch.autograd.grad(reference, a)[0])

    def test_weighted_loss_has_declared_value_and_gradient_priority(self):
        prediction = torch.ones((2, 4), requires_grad=True)
        target = torch.zeros_like(prediction)
        loss = weighted_mse(prediction, target, [32., 1., 1., 1.])
        self.assertAlmostEqual(float(loss.detach()), 1.)
        gradient = torch.autograd.grad(loss, prediction)[0]
        self.assertAlmostEqual(float(gradient[0, 0] / gradient[0, 1]), 32.)
        for weights in ([0., 1., 1., 1.], [float("nan"), 1., 1., 1.], [1., 2.]):
            with self.assertRaises(ValueError):
                weighted_mse(prediction, target, weights)

    def test_config_reserves_new_ids_and_declares_full_snapshot_pool(self):
        self.assertEqual(validate_config(config()), config())
        for key, value in (("test_layout_ids", [440, 441]), ("test_layout_ids", [500, 500]),
                           ("epochs", True), ("snapshot_epochs", [1]), ("mb_error_weight", float("nan"))):
            invalid = dict(config(), **{key: value})
            with self.assertRaises(ValueError):
                validate_config(invalid)

    def test_physical_selection_prefers_constraint_passes_and_keeps_seed_groups(self):
        refs, rows = {}, []
        for loss in ("uniform", "weighted"):
            for seed in (7, 17):
                for rule in ("mse", "snapshot"):
                    name = f"{loss}_{seed}_{rule}"
                    refs[name] = {"loss": loss, "seed": seed, "epoch": 2 if rule == "mse" else 1,
                                  "validation_mse": .001 if rule == "mse" else .01, "is_mse_best": rule == "mse"}
                    # A high-rate failed proposal must never outrank a lower-rate passing one.
                    for case in range(2):
                        passed = rule == "snapshot" or case == 0
                        if loss == "weighted" and seed == 17:
                            passed = False
                        rows.append({"candidate": name, "audit_pass": passed,
                                     "gated_bob_rate": 4. if passed else 0.})
        before = deepcopy(refs)
        selected = select_checkpoints(refs, rows)
        self.assertEqual(selected["chosen_group"], "uniform_physics")
        self.assertEqual(selected["selected"]["uniform_physics_s7"]["epoch"], 1)
        self.assertEqual(selected["selected"]["uniform_mse_s7"]["epoch"], 2)
        self.assertEqual(len(selected["selected"]), 8)
        self.assertFalse(selected["test_used_for_selection"])
        self.assertEqual(before, refs)
        with self.assertRaises(ValueError):
            select_checkpoints(refs, [])

    def test_physical_selection_uses_gated_rate_to_break_pass_count_ties(self):
        refs, rows = {}, []
        for loss in ("uniform", "weighted"):
            for k in range(2):
                name = f"{loss}_{k}"
                refs[name] = {"loss": loss, "seed": 7, "epoch": k + 1,
                              "validation_mse": .001 + k, "is_mse_best": k == 0}
                rows.append({"candidate": name, "audit_pass": True, "gated_bob_rate": 2. + k})
        result = select_checkpoints(refs, rows)
        self.assertEqual(result["selected"]["weighted_physics_s7"]["candidate"], "weighted_1")
        self.assertEqual(result["chosen_group"], "uniform_physics")  # declared tie rule

    def test_matched_data_rejects_cross_split_and_duplicate_cases(self):
        def arrays(lid):
            return {"node": np.zeros((2, 3, 15)), "global_features": np.zeros((2, 24)),
                    "edge_index": np.zeros((2, 2, 6), dtype=int), "edge_attr": np.zeros((2, 6, 2)),
                    "labels": np.tile([[[.2, .3, .5, .4]]], (2, 3, 1)),
                    "layout_id": np.array([lid, lid]), "xi": np.array([.75, .8]),
                    "total_power_dbm": np.array([30., 30.])}
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.savez(root / "matched_train.npz", **arrays(400))
            np.savez(root / "matched_validation.npz", **arrays(424))
            data = load_matched(root, 2, {"train": [400], "validation": [424]})
            self.assertEqual(data["train"][TARGET_KEY].shape, (2, 4))
            with self.assertRaises(ValueError):
                load_matched(root, 2, {"train": [400], "validation": [400]})
            duplicate = arrays(400)
            duplicate["xi"][:] = .75
            np.savez(root / "matched_train.npz", **duplicate)
            with self.assertRaises(ValueError):
                load_matched(root, 2, {"train": [400], "validation": [424]})

    def test_real_training_checkpoint_replays_and_losses_share_initial_weights(self):
        rng = np.random.default_rng(5)
        data = {"node": rng.normal(size=(4, 3, 15)).astype("f4"),
                "global_features": rng.normal(size=(4, 24)).astype("f4"),
                "edge_index": np.tile(np.array([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]), (4, 1, 1)),
                "edge_attr": np.zeros((4, 6, 2), dtype="f4"),
                TARGET_KEY: np.tile([.2, .3, .5, .4], (4, 1))}
        c = {"models": ["gnn"], "seeds": [7], "epochs": 2, "batch_size": 2, "learning_rate": .001,
             "weight_decay": 1e-5, "patience": 1, "min_delta": 1e-7, "gnn_hidden": 4, "gnn_layers": 1,
             "mlp_hidden": 4, "mlp_layers": 1, "gradient_clip": 5., "threads": 1, "notes": "test"}
        plan = {"training_config": c, "loss_weights": {"uniform": [1., 1., 1., 1.], "weighted": [32., 1., 1., 1.]},
                "snapshot_epochs": [1, 2], "feature_schema": {}, "normalization": {},
                "sources_sha256": {"previous_inputs": {}}, "fit_plan_sha256": "fixture"}
        bounds = {"mb": [.05, .49], "s": [.05, .95]}
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fits = []
            for loss in plan["loss_weights"]:
                folder = root / loss
                result = train_fit(folder, loss, plan, bounds, {"train": data, "validation": data}, loss, 7)
                self.assertEqual(result["epochs_completed"], 2)
                self.assertFalse(result["early_stopping"])
                self.assertEqual(len(result["candidates"]), 3)
                model, checkpoint = load_checkpoint(folder / "best_model.pt")
                actual = np.mean((predict(model, data) - data[TARGET_KEY]) ** 2)
                self.assertAlmostEqual(actual, checkpoint["validation_mse"], places=12)
                self.assertFalse(checkpoint["test_used_for_selection"])
                self.assertEqual(sha256(folder / "history.csv"), read_json(folder / "fit_manifest.json")["history_sha256"])
                fits.append(result)
            self.assertEqual(fits[0]["initial_state_sha256"], fits[1]["initial_state_sha256"])


if __name__ == "__main__":
    unittest.main()
