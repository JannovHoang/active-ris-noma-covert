"""Step 5b contracts: information flow, symmetry, gradients, checkpointing,
reproducibility, and the original/SIC mapping at the physics boundary.
"""
from copy import deepcopy
from pathlib import Path
import contextlib
import io
import tempfile
import unittest
import numpy as np
from ris_noma.learning_data import INPUT_KEYS, TARGET_KEY, physics_allocation
from ris_noma.config import load_config
from ris_noma.geometry import sample_layout
from ris_noma.channels import build_bank, public_order
from ris_noma.random_streams import make_rng
from ris_noma.graph_data import statistical_graph, allocation_head
from ris_noma.evaluator import evaluate_with_dep
from ris_noma.validation import statistical_screen
from ris_noma.learning_physics import assess_allocation
try:
    import torch
    from ris_noma.learning_models import AllocationGNN, AllocationMLP, PowerHead
    from ris_noma.learning_train import (configure_runtime, tensors, predict, train_one,
                                         load_checkpoint, validate_training_config)
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

ROOT = Path(__file__).resolve().parents[1]
BOUNDS = {"mb": [.05, .49], "s": [.05, .95]}


def training_fixture(n=10):
    rng = np.random.default_rng(49)
    pairs = np.array([(i,j) for i in range(3) for j in range(3) if i != j]).T
    node, glob = rng.normal(size=(n,3,15)), rng.normal(size=(n,24))
    x = glob[:, 0]
    mb = .15 + .04 * np.tanh(x)
    s = .4 + .1 * np.tanh(x)
    share = .55 + .05 * np.tanh(node[:,1,0])
    y = np.column_stack([mb, (1-mb)*share, (1-mb)*(1-share), s])
    return {"node": node, "global_features": glob, "edge_index": np.repeat(pairs[None], n, axis=0),
            "edge_attr": np.repeat(np.eye(2)[[0,0,1,0,1,1]][None], n, axis=0), TARGET_KEY: y,
            "layout_id": np.arange(n), "xi": np.full(n,.9), "bob_reference_rate_label": np.ones(n)}


@unittest.skipIf(torch is None, "Install requirements_step5b.txt to test PyTorch learning")
class TestLearningModels(unittest.TestCase):
    def setUp(self):
        configure_runtime(4)
        self.data = training_fixture()
        self.t = tensors(self.data, "cpu")

    def model_inputs(self):
        return {k: self.t[k] for k in INPUT_KEYS}

    def test_torch_head_matches_numpy_and_backpropagates(self):
        raw = torch.tensor([[.7, -.3, .6, 1.2]], requires_grad=True)
        y = PowerHead(BOUNDS)(raw[:,0], raw[:,1:3], raw[:,3])
        expected = allocation_head(.7, [-.3,.6], 1.2, BOUNDS)
        np.testing.assert_allclose(y.detach().numpy()[0], expected, atol=1e-7)
        y.square().sum().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertGreater(float(raw.grad.abs().sum()), 0)

    def test_both_models_enforce_bounds_and_simplex(self):
        for model in [AllocationGNN(BOUNDS,16,2), AllocationMLP(BOUNDS,16,2)]:
            y = model(**self.model_inputs()).detach().numpy()
            np.testing.assert_allclose(y[:,:3].sum(axis=1), 1, atol=1e-7)
            self.assertTrue(np.all(y[:,0] >= .05) and np.all(y[:,0] <= .49))
            self.assertTrue(np.all(y[:,3] >= .05) and np.all(y[:,3] <= .95))
            self.assertTrue(np.all(y[:,1:3] >= 0))

    def test_gnn_is_equivariant_to_public_identity_swap(self):
        model = AllocationGNN(BOUNDS,16,3).eval()
        original = model(**self.model_inputs()).detach().numpy()
        perm = torch.tensor([0,2,1])
        inputs = self.model_inputs()
        inputs["node"] = inputs["node"][:,perm]
        inputs["edge_index"] = perm[inputs["edge_index"]]
        swapped = model(**inputs).detach().numpy()
        np.testing.assert_allclose(original[:,[0,2,1,3]], swapped, atol=2e-7)

    def test_edge_list_reordering_does_not_change_either_model(self):
        for model in [AllocationGNN(BOUNDS,16,2), AllocationMLP(BOUNDS,16,2)]:
            inputs = self.model_inputs()
            original = model(**inputs).detach().numpy()
            p = [5,2,0,4,1,3]
            inputs["edge_index"] = inputs["edge_index"][:,:,p]
            inputs["edge_attr"] = inputs["edge_attr"][:,p]
            np.testing.assert_allclose(original, model(**inputs).detach().numpy(), atol=2e-7)

    def test_gnn_receives_relation_attributes_and_all_message_layers_get_gradients(self):
        model = AllocationGNN(BOUNDS,16,3)
        inputs = self.model_inputs()
        attributes = inputs["edge_attr"].clone().requires_grad_()
        inputs["edge_attr"] = attributes
        loss = (model(**inputs) - self.t[TARGET_KEY]).square().mean()
        loss.backward()
        self.assertGreater(float(attributes.grad.abs().sum()), 0)
        for layer in model.layers:
            self.assertTrue(all(p.grad is not None for p in layer.parameters()))
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in layer.message.parameters()), 0)

    def test_predict_ignores_reference_labels_and_provenance_fields(self):
        model = AllocationGNN(BOUNDS,16,2)
        before = predict(model, self.data)
        changed = deepcopy(self.data)
        changed[TARGET_KEY][:] = 999
        changed["layout_id"][:] = -1
        changed["bob_reference_rate_label"][:] = 1e100
        changed["audit_result"] = np.zeros(len(changed[TARGET_KEY]))
        np.testing.assert_array_equal(before, predict(model, changed))


@unittest.skipIf(torch is None, "Install requirements_step5b.txt to test training")
class TestTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        cls.temp = tempfile.TemporaryDirectory()
        cls.c = json.loads((ROOT / "configs/training_pilot_5b.json").read_text())
        cls.c.update(epochs=50,patience=50,batch_size=10,gnn_hidden=8,gnn_layers=1,mlp_hidden=16,mlp_layers=1)
        cls.train, cls.validation = training_fixture(), training_fixture(4)
        cls.results = {}
        with contextlib.redirect_stdout(io.StringIO()):
            for name in ["one", "two"]:
                cls.results[name] = train_one("mlp",7,cls.c,BOUNDS,cls.train,cls.validation,{}, {}, {},Path(cls.temp.name)/name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_training_decreases_loss_and_best_checkpoint_has_validation_minimum(self):
        import csv
        r = self.results["one"]
        self.assertLess(r["train"]["mse"], r["initial_train_mse"])
        with (Path(self.temp.name)/"one/history.csv").open() as f:
            values = [float(row["validation_mse"]) for row in csv.DictReader(f)]
        self.assertAlmostEqual(r["validation"]["mse"], min([r["initial_validation_mse"], *values]), places=10)
        self.assertFalse(r["test_used_for_selection"])

    def test_seed_reproduces_training_and_checkpoint_predictions(self):
        one, two = self.results["one"], self.results["two"]
        self.assertEqual(one["best_epoch"], two["best_epoch"])
        self.assertEqual(one["validation"], two["validation"])
        a, _ = load_checkpoint(Path(self.temp.name)/"one/best_model.pt")
        b, _ = load_checkpoint(Path(self.temp.name)/"two/best_model.pt")
        np.testing.assert_array_equal(predict(a,self.validation),predict(b,self.validation))

    def test_weights_only_checkpoint_round_trip_preserves_metrics(self):
        model, checkpoint = load_checkpoint(Path(self.temp.name)/"one/best_model.pt")
        prediction = predict(model,self.validation)
        self.assertAlmostEqual(float(np.mean((prediction-self.validation[TARGET_KEY])**2)),self.results["one"]["validation"]["mse"],places=12)
        self.assertEqual(checkpoint["selection_metric"],"validation_allocation_mse")

    def test_config_rejects_duplicate_seeds_and_nonpositive_learning_rate(self):
        for key,value in [("seeds",[7,7]),("learning_rate",0),("models",["gnn","x"]),("epochs",0)]:
            c=deepcopy(self.c); c[key]=value
            with self.assertRaises(ValueError): validate_training_config(c)


class TestPhysicsBoundary(unittest.TestCase):
    def test_float32_rounding_is_corrected_without_repairing_invalid_predictions(self):
        y = np.array([.2,.3,.5,.6],np.float32)
        fixed = physics_allocation(y,BOUNDS)
        self.assertAlmostEqual(fixed[:3].sum(),1,places=15)
        for bad in [[.9,.05,.05,.5],[.2,.5,.5,.5],[.2,-.1,.9,.5],[.2,.3,.5,float('nan')]]:
            with self.assertRaises(ValueError): physics_allocation(bad,BOUNDS)

    def test_original_identity_power_is_mapped_to_frozen_sic_order(self):
        c=load_config(ROOT/"configs/legacy_physics_k2_pilot.json")
        c.update(n_fading=180,layout_id=100)
        layout=sample_layout(c,make_rng(7,100,"geometry"))
        bank=build_bank(c,layout,make_rng(7,100,"label"))
        order=np.array([1,0]); y=np.array([.2,.3,.5,.6])
        summary,screen=assess_allocation(c,layout,bank,order,y,.8,1,.05)
        direct,samples,_=evaluate_with_dep(c,layout,bank,order,.2,[.5,.3],.6,.8,True)
        self.assertEqual(summary["allocation_sic_order"],direct["allocation_sic_order"])
        self.assertAlmostEqual(summary["dep_min"],direct["dep_min"])
        np.testing.assert_array_equal(summary["public_mean_rate_bit_s_hz"],direct["public_mean_rate_bit_s_hz"])
        self.assertEqual(screen,statistical_screen(c,direct,samples,order,.8,1,.05))


if __name__ == "__main__":
    unittest.main()
