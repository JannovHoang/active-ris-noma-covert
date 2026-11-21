"""Statistical feature provenance, original/SIC mapping and grouped export."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import contextlib
import io
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import run_step3b
import run_step4a
import run_step4b
import run_step5a
from ris_noma.config import load_config
from ris_noma.geometry import sample_layout
from ris_noma.channels import build_bank, public_order
from ris_noma.random_streams import make_rng
from ris_noma.pso import PSOSettings
from ris_noma.graph_data import (statistical_graph, gain_statistics, original_allocation,
                                allocation_head, fit_normalizer, transform_graphs,
                                NODE_FEATURES, GLOBAL_FEATURES)
from ris_noma.dataset import export_dataset, check_dataset, validate_partition

ROOT = Path(__file__).resolve().parents[1]


class TestStatisticalGraph(unittest.TestCase):
    def setUp(self):
        self.c = load_config(ROOT / "configs/legacy_physics_k2_pilot.json")
        self.c.update(n_fading=120, layout_id=100)
        self.layout = sample_layout(self.c, make_rng(7, 100, "geometry"))
        self.bank = build_bank(self.c, self.layout, make_rng(7, 100, "label"))
        self.order = public_order(self.bank)

    def test_schema_is_finite_and_edges_are_sic_interference_relations(self):
        g = statistical_graph(self.c, self.layout, self.bank, self.order, .9)
        self.assertEqual(g["node_raw"].shape, (3, 15))
        self.assertEqual(g["global_raw"].shape, (24,))
        self.assertEqual(g["edge_index"].shape, (2, 6))
        np.testing.assert_array_equal(g["edge_attr"].sum(axis=0), [3., 3.])
        np.testing.assert_array_equal(g["edge_attr"].sum(axis=1), np.ones(6))
        self.assertTrue(all(np.all(np.isfinite(v)) for v in g.values()))

    def test_only_xi_changes_across_targets_and_reference_allocation_is_not_a_feature(self):
        a = statistical_graph(self.c, self.layout, self.bank, self.order, .75)
        c = deepcopy(self.c);c["reference_allocation"] = {"mb": .2, "public": [.3, .5], "s": .6}
        b = statistical_graph(c, self.layout, self.bank, self.order, .95)
        np.testing.assert_array_equal(a["node_raw"], b["node_raw"])
        np.testing.assert_array_equal(a["global_raw"][1:], b["global_raw"][1:])
        self.assertEqual(a["global_raw"][0], .75)
        self.assertEqual(b["global_raw"][0], .95)
        with self.assertRaises(TypeError):
            statistical_graph(c, self.layout, self.bank, self.order, .9, allocation=[.2,.3,.5,.6])

    def test_public_user_permutation_preserves_features_and_relations(self):
        a = statistical_graph(self.c, self.layout, self.bank, self.order, .9)
        cols = [0, 2, 1, 3]
        bbank = replace(self.bank, h_ru=self.bank.h_ru[:, cols], h_tilde=self.bank.h_tilde[:, cols],
                        g_scalar_tilde=self.bank.g_scalar_tilde[:, cols],
                        signal_gain_tilde=self.bank.signal_gain_tilde[:, cols], noise_gain_tilde=self.bank.noise_gain_tilde[:, cols])
        layout = replace(self.layout, public_xy=self.layout.public_xy[::-1], receiver_xy=self.layout.receiver_xy[cols],
                         distance_ru_m=self.layout.distance_ru_m[cols], loss_ru=self.layout.loss_ru[cols])
        c = deepcopy(self.c)
        for key in ["qos_public", "target_public"]:c["requirements"][key] = c["requirements"][key][::-1]
        c["noise_dbm"]["public"] = c["noise_dbm"]["public"][::-1]
        b = statistical_graph(c, layout, bbank, 1-self.order, .9)
        permutation = np.array([0, 2, 1])
        np.testing.assert_allclose(a["node_raw"][permutation], b["node_raw"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(a["global_raw"], b["global_raw"], rtol=1e-12, atol=1e-12)
        old_edges = {tuple(permutation[e]): f.tolist() for e, f in zip(a["edge_index"].T, a["edge_attr"])}
        new_edges = {tuple(e): f.tolist() for e, f in zip(b["edge_index"].T, b["edge_attr"])}
        self.assertEqual(old_edges, new_edges)

    def test_original_target_and_pso_coordinates_round_trip_with_reversed_sic(self):
        y, z = original_allocation({"mb": .2, "public": [.5, .3], "s": .6}, [1,0], self.c["bounds"])
        np.testing.assert_allclose(y, [.2,.3,.5,.6])
        self.assertAlmostEqual(z[1], .625)

    def test_output_head_enforces_bounds_simplex_and_public_permutation(self):
        for value in [-1000., -2., 0., 2., 1000.]:
            y = allocation_head(value, [value, -value], -value, self.c["bounds"])
            self.assertTrue(np.all(np.isfinite(y)))
            self.assertAlmostEqual(y[:3].sum(), 1.)
            self.assertTrue(.05 <= y[0] <= .49 and .05 <= y[3] <= .95)
            swapped = allocation_head(value, [-value, value], -value, self.c["bounds"])
            np.testing.assert_allclose(y[[0,2,1,3]], swapped)

    def test_normalizer_uses_only_training_data_and_shared_node_scaling(self):
        node = np.arange(90.).reshape(2,3,15)
        glob = np.arange(48.).reshape(2,24)
        normalizer = fit_normalizer(node, glob)
        n, g = transform_graphs(node, glob, normalizer)
        np.testing.assert_allclose(n.mean(axis=(0,1)), 0, atol=1e-12)
        np.testing.assert_allclose(g.mean(axis=0), 0, atol=1e-12)
        shifted, _ = transform_graphs(node + 1000, glob, normalizer)
        self.assertGreater(float(shifted.mean()), 10.)
        permuted, _ = transform_graphs(node[:,[2,0,1]], glob, normalizer)
        np.testing.assert_allclose(n[:,[2,0,1]], permuted)

    def test_gain_statistics_handles_constant_gains_without_nan(self):
        s = gain_statistics(np.ones((5,4)), np.ones((5,4)))
        np.testing.assert_array_equal(s, np.zeros((4,5)))

    def test_partition_rejects_overlap_previously_examined_test_and_final_test_ids(self):
        for config, ids in [
            ({"purpose":"pilot","splits":{"train":[100],"validation":[100],"test":[]}},[100]),
            ({"purpose":"pilot","splits":{"train":[100],"validation":[1],"test":[]}},[100,1]),
            ({"purpose":"pilot","splits":{"train":[100],"validation":[],"test":[10000]}},[100,10000]),
        ]:
            with self.assertRaises(ValueError):validate_partition(config, ids)


class TestDatasetPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)
        cls.pipeline = cls.base / "pilot"
        c = load_config(ROOT / "configs/legacy_physics_k2_pilot.json")
        c["requirements"].update(xi=[.2,.4], qos_public=[.01,.01], target_public=[.01,.01], target_bob=.01)
        policy = run_step3b.load_policy(ROOT / "configs/validation_pilot.json")
        policy.update(label_samples=150, validation_samples=1500, heldout_samples=1500, margins=[0.,.02])
        cls.partition = {"purpose":"pilot","splits":{"train":[100],"validation":[101],"test":[102]},
                         "pso_seeds":[1],"study_alpha":.05,"heldout_seed_new":2}
        cls.c, cls.settings, cls.policy = c, PSOSettings(swarm_size=4,sweeps=2), policy
        with contextlib.redirect_stdout(io.StringIO()):
            cls.status = run_step5a.generate(c, cls.settings, policy,
                                            cls.partition, cls.pipeline)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_generator_freezes_partition_and_exports_complete_disjoint_graphs(self):
        self.assertEqual(self.status["checks"], "passed")
        self.assertTrue(self.status["ready_for_training_pilot"])
        self.assertEqual(self.status["labeled_graphs"], 6)
        plan = run_step4a.read_json(self.pipeline / "dataset_plan.json")
        self.assertEqual(plan["partition"], self.partition)
        for role, lid in [("train",100),("validation",101),("test",102)]:
            with np.load(self.pipeline / "graphs" / f"{role}.npz",allow_pickle=False) as d:
                self.assertEqual(set(d["layout_id"]), {lid})
                self.assertEqual(d["node"].shape, (2,3,len(NODE_FEATURES)))
                self.assertEqual(d["global_features"].shape, (2,len(GLOBAL_FEATURES)))
        normalizer = run_step4a.read_json(self.pipeline / "graphs/normalization.json")
        self.assertEqual(normalizer["training_layout_ids"], [100])

    def test_export_does_not_modify_source_and_resume_does_not_rebuild_features(self):
        source = self.pipeline / "selection4b"
        before = {str(p):run_step4a.sha(p) for p in source.rglob('*') if p.is_file()}
        output = self.base / "again"
        with contextlib.redirect_stdout(io.StringIO()):
            export_dataset(source, output, self.partition)
            with patch("ris_noma.dataset.build_bank",side_effect=AssertionError("Must skip completed export")):
                result = export_dataset(source, output, self.partition,resume=True)
        self.assertEqual(result["labeled_graphs"],6)
        self.assertEqual(before, {str(p):run_step4a.sha(p) for p in source.rglob('*') if p.is_file()})

    def test_complete_generator_resume_reuses_all_three_finished_stages(self):
        with contextlib.redirect_stdout(io.StringIO()), \
             patch("run_step3b.build_bank",side_effect=AssertionError("Must skip completed PSO stage")), \
             patch("run_step4b.build_bank",side_effect=AssertionError("Must skip completed audit stage")), \
             patch("ris_noma.dataset.build_bank",side_effect=AssertionError("Must skip completed graph export")):
            result=run_step5a.generate(self.c,self.settings,self.policy,self.partition,self.pipeline,resume=True)
        self.assertEqual(result['checks'],'passed')
        self.assertEqual(result['labeled_graphs'],6)

    def changed_source(self, name):
        destination = self.base / name
        shutil.copytree(self.pipeline / "selection4b",destination)
        return destination

    def rewrite_child(self, source, lid, rows, frozen):
        folder=source/f"layout_{lid:06d}"
        run_step3b.write_json(folder/'selected_by_xi.json',rows)
        run_step3b.write_csv(folder/'selected_by_xi.csv',rows)
        run_step3b.write_json(folder/'selection_frozen.json',frozen)
        state=run_step4a.read_json(folder/'status.json')
        state['selection_sha256_before_and_after_audit']=run_step4a.sha(folder/'selection_frozen.json')
        run_step3b.write_json(folder/'status.json',state)
        run_step4a.finalize_job(folder,0.)

    def test_failed_heldout_audit_is_retained_as_a_label_without_filtering(self):
        source=self.changed_source('audit_review_source')
        folder=source/'layout_000100'
        rows=run_step4a.read_json(folder/'selected_by_xi.json')
        frozen=run_step4a.read_json(folder/'selection_frozen.json')
        rows[0].update(status='selected_on_validation_audit_review',heldout_bound_checks_pass=False)
        self.rewrite_child(source,100,rows,frozen)
        with contextlib.redirect_stdout(io.StringIO()):
            result=export_dataset(source,self.base/'audit_review_dataset',self.partition)
        self.assertEqual(result['labeled_graphs'],6)
        self.assertEqual(result['counts']['train']['heldout_review_labels_retained'],1)

    def test_missing_label_is_recorded_without_a_fabricated_target(self):
        source=self.changed_source('missing_source')
        folder=source/'layout_000100'
        rows=run_step4a.read_json(folder/'selected_by_xi.json')
        frozen=run_step4a.read_json(folder/'selection_frozen.json')
        rows[0].update(proposal_id=None,status='no_candidate_passed_validation',heldout_bound_checks_pass=None)
        frozen['selection'][0]['proposal_id']=None
        self.rewrite_child(source,100,rows,frozen)
        with contextlib.redirect_stdout(io.StringIO()):
            result=export_dataset(source,self.base/'missing_dataset',self.partition)
        self.assertEqual(result['labeled_graphs'],5)
        self.assertEqual(result['counts']['train']['missing'],1)
        records=[json.loads(line) for line in (self.base/'missing_dataset/records.jsonl').read_text().splitlines()]
        self.assertEqual(len(records),6)
        self.assertNotIn('allocation_original_order',next(r for r in records if not r['label_available']))

    def test_tampered_export_is_rejected_and_existing_output_is_protected(self):
        copied=self.base/'tampered'
        shutil.copytree(self.pipeline/'graphs',copied)
        with (copied/'train.npz').open('ab') as f:f.write(b'tampered')
        with self.assertRaisesRegex(ValueError,'output changed'):check_dataset(copied)
        with self.assertRaisesRegex(ValueError,'Nonempty dataset'):
            export_dataset(self.pipeline/'selection4b',self.pipeline/'graphs',self.partition)


if __name__ == '__main__':unittest.main()
