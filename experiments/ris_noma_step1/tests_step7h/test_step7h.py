"""Scientific/operational invariants of 7h, without needing the research datasets."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import step7h
import numpy as np
import torch

from step7h.common import (METHODS, write, read, commit, committed, digest, verify, safe_path, object_hash)
from step7h.settings import DEFAULT, settings, smoke_settings
from step7h.physics import entropy, context, corrected
from step7h.benchmark import run_search, expected_keys, variants
from step7h.analysis import metrics, paired, budget_selection, layout_intervals
from step7h.checks import verify_decision
from ris_noma.pso import Outcome, PSOSettings, optimize
from ris_noma.learning_train import configure_runtime
from ris_noma.channels import public_order
from ris_noma.multik_core import MultiKObjective, original_allocation
from ris_noma.multik_graph import statistical_graph, fit_normalizer, pack_graphs
from ris_noma.multik_models import predict
from ris_noma.multik_learning_models import build_model
from ris_noma.stable_assessment import raw_assessment
from ris_noma.multik_data_core import at_target


class ToyObjective:
    def __init__(self, feasible=True): self.feasible = feasible; self.points = []
    def evaluate(self, z, xi):
        self.points.append(z.copy())
        return Outcome(float(-np.square(z-.3).sum()), 0. if self.feasible else .1,
                       self.feasible, {"dep_min": .9})


def observation(lid, method, passed=True, budget=0, replicate=7):
    return {"case_id":f"case_{lid}","layout_id":lid,"n_public":2,"regime":"seen_k",
        "replicate":replicate,"method":method,"budget":budget,"passed":passed,
        "bob_rate":4.,"gated_bob_rate":4. if passed else 0.,"online_seconds":.01,
        "pso_seconds":.005 if budget else 0.,"objective_evaluations":budget,
        "invalid_layout_assumption":False,"failed_checks":[] if passed else ["covertness_gamma"]}


class IOAndProtocolTests(unittest.TestCase):
    def test_bundled_physics_runtime_is_byte_identical_to_frozen_manifest(self):
        root=Path(__file__).resolve().parents[1]/"step7h"
        manifest=read(root/"frozen_runtime_manifest.json")
        self.assertEqual(verify(root/"frozen_runtime",manifest),manifest["unchanged_files"])

    def test_exactly_six_methods_no_ridge(self):
        self.assertEqual(len(METHODS),6)
        self.assertIn("gnn_direct",METHODS);self.assertIn("mlp_direct",METHODS)
        self.assertFalse(any("ridge" in x for x in METHODS))

    def test_main_budget_or_selector_change_rejected(self):
        for key,value in (("full_budget",1770),("final_selection","independent_bank"),
                          ("fallback","full_pso"),("training_prefix",64)):
            c=deepcopy(DEFAULT);c[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):settings(c)

    def test_smoke_cannot_use_reserved_test_ids(self):
        c=smoke_settings();c["smoke_test_layout_ids"]=[5000]
        with self.assertRaises(ValueError):settings(c)

    def test_seed_replicates_must_be_distinct(self):
        c=smoke_settings();c["replicates"]*=2
        with self.assertRaises(ValueError):settings(c)

    def test_utf8_json_independent_of_windows_default_encoding(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/"Hồ_sơ.json";data={"ghi_chú":"Kiểm định độc lập, ξ và độ dư"}
            write(path,data)
            self.assertEqual(read(path),data)

    def test_atomic_receipt_detects_changed_bytes(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/"record.json";commit(path,{"x":1})
            self.assertEqual(committed(path),{"x":1})
            write(path,{"x":2})
            with self.assertRaises(ValueError):committed(path)

    def test_incomplete_unreceipted_record_can_be_recomputed(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/"record.json";write(path,{"x":1})
            self.assertIsNone(committed(path))

    def test_portable_manifest_and_path_traversal_rejection(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);write(root/"child/file.json",{"x":1})
            self.assertEqual(verify(root,{"sha256":{"child\\file.json":digest(root/"child/file.json")}}),1)
            for name in ("../secret","C:\\secret","/secret"):
                with self.subTest(name=name),self.assertRaises(ValueError):safe_path(root,name)


class SearchBudgetTests(unittest.TestCase):
    def test_actual_call_count_includes_initial_swarm(self):
        for sweeps in (1,5,10,15):
            objective=ToyObjective()
            result=run_search(objective,.9,2,PSOSettings(sweeps=sweeps),np.random.default_rng(7))
            self.assertEqual(result.evaluations,30*sweeps)
            self.assertEqual(len(objective.points),30*sweeps)
            self.assertEqual(result.history[0]["evaluations"],30)

    def test_one_injected_prediction_and_same_29_random_particles(self):
        random,warm=ToyObjective(),ToyObjective();center=np.array([[.4,.5,.6]])
        cfg=PSOSettings(sweeps=1)
        run_search(random,.9,2,cfg,np.random.default_rng(101))
        run_search(warm,.9,2,cfg,np.random.default_rng(101),center)
        np.testing.assert_array_equal(warm.points[0],center[0])
        np.testing.assert_array_equal(warm.points[1:30],random.points[1:30])

    def test_budget_specific_inertia_is_not_a_long_run_prefix(self):
        a=run_search(ToyObjective(),.9,2,PSOSettings(sweeps=5),np.random.default_rng(7))
        b=run_search(ToyObjective(),.9,2,PSOSettings(sweeps=15),np.random.default_rng(7))
        self.assertEqual(a.history[1]["inertia_used_for_update"],.9)
        self.assertAlmostEqual(a.history[-1]["inertia_used_for_update"],.4)
        self.assertNotEqual(a.history[2]["inertia_used_for_update"],b.history[2]["inertia_used_for_update"])

    def test_infeasible_short_search_does_not_run_a_full_fallback(self):
        objective=ToyObjective(False)
        result=run_search(objective,.9,2,PSOSettings(sweeps=5),np.random.default_rng(7))
        self.assertFalse(result.outcome.feasible);self.assertEqual(len(objective.points),150)

    def test_declared_product_contains_both_direct_endpoints(self):
        p={"settings":smoke_settings(),"short_budgets":[30,60]}
        job={"cases":[{"case_index":0}]}
        self.assertEqual(len(expected_keys(p,job)),9)
        self.assertIn((0,7,"gnn_direct",0),expected_keys(p,job))


class StatisticalAnalysisTests(unittest.TestCase):
    def test_absolute_intervals_resample_layouts_and_refuse_single_cluster_certainty(self):
        rows=[observation(i,"gnn_direct") for i in (1,2,3)]
        result=layout_intervals(rows,200,7)
        self.assertEqual(result["layout_clusters"],3)
        self.assertEqual(result["pass_fraction_ci95"],[1.,1.])
        self.assertIsNone(layout_intervals(rows[:1],200,7)["pass_fraction_ci95"])

    def test_geometry_failure_stays_in_primary_denominator(self):
        a=observation(1,"gnn_direct");b=observation(2,"gnn_direct",False)
        b["invalid_layout_assumption"]=True
        result=metrics([a,b])
        self.assertEqual(result["pass_fraction"],.5)
        self.assertEqual(result["gated_rate"],2.)
        self.assertEqual(result["eligible_pass_fraction"],1.)

    def test_identical_methods_have_zero_paired_gap_and_three_clusters(self):
        left=[observation(i,"gnn_short_pso",budget=150) for i in (1,2,3)]
        right=[observation(i,"random_short_pso",budget=150) for i in (1,2,3)]
        result=paired(left,right,200,7317)
        self.assertEqual(result["layout_clusters"],3)
        self.assertEqual(result["pass_difference_ci95"],[0.,0.])
        self.assertEqual(result["gated_rate_difference"],0.)

    def test_missing_pair_or_duplicate_is_an_error(self):
        a=[observation(i,"gnn_direct") for i in (1,2)]
        b=[observation(1,"mlp_direct")]
        with self.assertRaises(ValueError):paired(a,b,200,7)
        with self.assertRaises(ValueError):paired(a+a,a+a,200,7)

    def test_budget_rule_requires_both_networks(self):
        c=smoke_settings();rows=[]
        for lid in (1,2):
            rows.append(observation(lid,"full_pso",budget=90))
            for budget in (30,60):
                rows.append(observation(lid,"gnn_short_pso",budget=budget))
                rows.append(observation(lid,"mlp_short_pso",passed=budget==60,budget=budget))
                rows.append(observation(lid,"random_short_pso",budget=budget))
            rows.extend(observation(lid,k) for k in ("gnn_direct","mlp_direct"))
        result=budget_selection(rows,c)
        self.assertEqual(result["selected_short_budget"],60)
        self.assertTrue(result["reference_target_met"])

    def test_no_budget_meets_target_records_failure_not_success(self):
        c=smoke_settings();rows=[observation(1,"full_pso",budget=90)]
        for budget in (30,60):
            rows.extend(observation(1,m,False,budget) for m in ("gnn_short_pso","mlp_short_pso","random_short_pso"))
        rows.extend(observation(1,m) for m in ("gnn_direct","mlp_direct"))
        result=budget_selection(rows,c)
        self.assertFalse(result["reference_target_met"])
        self.assertEqual(result["selected_short_budget"],60)

    def test_all_methods_failing_is_not_a_successful_quality_gate(self):
        c=smoke_settings();rows=[observation(1,"full_pso",False,budget=90)]
        for budget in (30,60):
            rows.extend(observation(1,m,False,budget) for m in ("gnn_short_pso","mlp_short_pso","random_short_pso"))
        rows.extend(observation(1,m,False) for m in ("gnn_direct","mlp_direct"))
        result=budget_selection(rows,c)
        self.assertFalse(result["reference_target_met"])
        self.assertFalse(result["reference_has_feasible_return"])


class PhysicsAndLeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_runtime(17,1,"cpu")
        cls.temp=TemporaryDirectory();cls.root=Path(cls.temp.name)
        fixture=read(Path(__file__).with_name("fixture.json"))
        for name,key in (("stage_plan","stage"),("physics","physics"),("dataset","dataset")):
            write(cls.root/f"inputs/{name}.json",fixture[key])
        cls.job={"id":"validation_l003000_k2","layout_id":3000,"n_public":2,"split":"validation","regime":"seen_k",
                 "cases":[{"case_index":0,"power_dbm":35.,"xi":.85}]}
        cls.plan={"stage":"development","settings":smoke_settings(),"jobs":[cls.job],
                  "certificate_pool_size":20,"certificate_alpha_per_group":.001}
        write(cls.root/"plan.json",cls.plan)
        cls.c,cls.layout,cls.bank,cls.meta=context(cls.root,cls.plan,cls.job,"development_input")
        cls.order=public_order(cls.bank)
        cls.graph=statistical_graph(cls.c,cls.layout,cls.bank,cls.order,.85)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_audit_barrier_precedes_channel_generation(self):
        with patch("step7h.physics.build_bank",side_effect=AssertionError("Must not generate audit bank")):
            with self.assertRaisesRegex(ValueError,"Freeze all proposals"):
                context(self.root,self.plan,self.job,"development_audit")

    def test_final_barrier_precedes_any_test_geometry(self):
        p=deepcopy(self.plan);p["stage"]="final"
        with patch("step7h.physics.geometry_from",side_effect=AssertionError("Must not create test geometry")):
            with self.assertRaisesRegex(ValueError,"Seal the development protocol"):
                context(self.root,p,self.job,"final_input")

    def test_job_outside_plan_rejected(self):
        j=dict(self.job,layout_id=5000)
        with self.assertRaises(ValueError):context(self.root,self.plan,j,"development_input")

    def test_random_bank_roles_have_distinct_streams(self):
        self.assertNotEqual(entropy(7,self.job,"development_input"),entropy(7,self.job,"development_audit"))
        self.assertNotEqual(entropy(7,self.job,"development_input"),entropy(7,self.job,"final_input"))

    def test_cached_objective_preserves_full_physical_rate_and_dep(self):
        y=np.r_[.2,.5,.3,.5]
        _,z=original_allocation(y[0],y[1:-1][self.order],y[-1],self.order,self.c["bounds"])
        cached=MultiKObjective(self.c,self.layout,self.bank,self.order).evaluate(z,.85)
        raw=raw_assessment(self.c,self.layout,self.bank,self.order,y,20,.001)
        self.assertAlmostEqual(cached.score,raw["bob_rate"],places=11)
        self.assertAlmostEqual(cached.metrics["dep_min"],raw["dep_gamma"],places=11)

    def test_zero_power_public_stream_is_rejected_by_physics(self):
        y=np.r_[.49,0.,.51,.5]
        raw=raw_assessment(self.c,self.layout,self.bank,self.order,y,20,.001)
        result=at_target(raw,.85)
        self.assertFalse(result["passed"])
        self.assertIn("positive_sic_ceiling_margins",result["failed_checks"])
        self.assertEqual(result["gated_bob_rate"],0.)

    def test_delta_preserves_simplex_public_ratios_and_s(self):
        y=np.r_[.2,.5,.3,.5];shift=corrected(y,.005,self.c["bounds"])
        self.assertAlmostEqual(shift[0],.195)
        self.assertAlmostEqual(shift[:-1].sum(),1.)
        self.assertAlmostEqual(shift[1]/shift[2],y[1]/y[2]);self.assertEqual(shift[-1],y[-1])

    def test_delta_clamps_at_declared_lower_bound(self):
        y=np.r_[.052,.6,.348,.5];shift=corrected(y,.005,self.c["bounds"])
        self.assertEqual(shift[0],.05)

    def test_certificate_check_rejects_relaxed_dep_bound(self):
        raw=raw_assessment(self.c,self.layout,self.bank,self.order,np.r_[.2,.5,.3,.5],20,.001)
        row={**raw,**at_target(raw,.85),"xi":.85,"returned":True,"dep_lower_margin":raw["dep_lower"]-.85}
        plan=dict(self.plan,certificate_pool_size=20,certificate_alpha_per_group=.001)
        verify_decision(row,plan,2)
        row["dep_lower"]+=.01
        with self.assertRaises(ValueError):verify_decision(row,plan,2)

    def test_normalizer_fit_changes_when_held_out_graph_is_included(self):
        held=deepcopy(self.graph);held["global_raw"][0]=.95
        norm=fit_normalizer([self.graph]);leaked=fit_normalizer([self.graph,held])
        self.assertNotEqual(norm,leaked)
        self.assertEqual(norm["training_graph_count"],1)

    def test_audit_metadata_is_not_consumed_by_models(self):
        normalizer=fit_normalizer([self.graph]);g=deepcopy(self.graph);g["audit_passed"]=True;g["label_rate"]=1e10
        a,b=pack_graphs([self.graph],normalizer),pack_graphs([g],normalizer)
        for name in a:np.testing.assert_array_equal(a[name],b[name])

    def test_both_models_support_variable_k_without_untrained_output_slots(self):
        job=dict(self.job,id="validation_l003000_k6",n_public=6)
        plan=dict(self.plan,jobs=[job])
        c,layout,bank,_=context(self.root,plan,job,"development_input")
        order=public_order(bank);g6=statistical_graph(c,layout,bank,order,.85)
        normalizer=fit_normalizer([self.graph])
        for kind in ("gnn","mlp"):
            model=build_model(kind,c["bounds"],{"hidden":32,"layers":2,"mlp_inner":112}).eval()
            single=predict(model,[self.graph],normalizer)[0]
            mixed=predict(model,[self.graph,g6],normalizer)
            np.testing.assert_allclose(single,mixed[0],rtol=0,atol=3e-7)
            self.assertEqual(len(mixed[1]),8)


if __name__=="__main__":unittest.main()
