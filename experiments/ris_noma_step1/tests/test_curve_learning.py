"""7g invariants: nested groups, small models, resume, frozen banks and paired curves."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from ris_noma import curve_data as data
from ris_noma.curve_fit import fit, read_fit
from ris_noma.curve_models import build_model, load_model
from ris_noma.curve_physics import audit_task, confidence, context, STREAMS
from ris_noma.curve_learning import select, initialize, seal, freeze_predictions
from ris_noma.curve_analysis import allocation_errors, physical_score, paired_bootstrap, table
from ris_noma.learning_data import read_json
from ris_noma.learning_train import configure_runtime
from ris_noma.multik_dataset import write_json
from ris_noma.multik_graph import fit_normalizer
from ris_noma.multik_models import predict
from test_continuous_learning import fixture, rows_for

ROOT=Path(__file__).resolve().parents[1]


def settings():
    return read_json(ROOT/"configs/learning_curve_smoke_7g.json")


def fit_plan(architecture="small"):
    c=settings();train=rows_for(2)+rows_for(3)
    cfg=data.fit_config(c,architecture)
    cfg.update(epochs=4,snapshot_epochs=[2,4])
    return {"config":cfg,"normalization":fit_normalizer([r["graph"] for r in train]),
            "plan_sha256":"fixed-unit-plan","training_public_counts":[2,3],
            "training_layout_ids":[2000],"architecture_name":architecture,
            "stage_completion_sha256":"fixed-unit-stage","corpus_plan_sha256":"fixed-unit-corpus"},train


class ConfigAndNestedDataTests(unittest.TestCase):
    def test_main_counts_and_model_sizes(self):
        c=data.validate_settings(read_json(ROOT/"configs/learning_curve_7g.json"))
        jobs=data.jobs_for(c,7)
        plan={"config":c,"jobs":jobs}
        self.assertEqual([sum(j["split"]=="train" for j in data.stage_jobs(plan,n))*8 for n in [32,64,128]],
                         [1024,2048,4096])
        self.assertEqual(sum(j["split"]=="validation" for j in jobs)*8,512)
        bounds=fixture()[0]["bounds"]
        for name in ("reference","small"):
            counts={kind:sum(p.numel() for p in build_model(kind,bounds,data.fit_config(c,name)).parameters())
                    for kind in ("gnn","mlp")}
            self.assertLess(abs(counts["gnn"]-counts["mlp"])/counts["gnn"],.01)
            self.assertEqual(counts["gnn"],122883 if name=="reference" else 24387)

    def test_prefixes_are_nested_and_validation_identical(self):
        c=settings();jobs=data.jobs_for(c,7);plan={"config":c,"jobs":jobs}
        small=data.stage_jobs(plan,2);large=data.stage_jobs(plan,4)
        self.assertTrue(all(j in large for j in small))
        self.assertEqual([j for j in small if j["split"]=="validation"],[j for j in large if j["split"]=="validation"])
        with self.assertRaises(ValueError):data.stage_jobs(plan,3)

    def test_reserved_test_ids_and_unseen_k_never_generate_jobs(self):
        c=settings()
        for job in data.jobs_for(c,7):
            self.assertNotIn(job["layout_id"],c["reserved_test_layout_ids"])
            self.assertNotIn(job["n_public"],c["reserved_unseen_public_counts"])
            self.assertNotEqual(job["split"],"test")

    def test_invalid_grouping_budget_or_nested_size_is_rejected(self):
        for mutate in ("overlap","sizes","k","budget","margin","probe","loss","unknown"):
            c=settings()
            if mutate=="overlap":c["reserved_test_layout_ids"]=[c["validation_layout_ids"][0]]
            elif mutate=="sizes":c["training_sizes"]=[2,3]
            elif mutate=="k":c["reserved_unseen_public_counts"]=[2]
            elif mutate=="budget":c["label"]["pso_evaluations"]=61
            elif mutate=="margin":c["label"]["search_dep_margin"]=.1
            elif mutate=="probe":c["physics"]["train_probe_layout_count"]=3
            elif mutate=="loss":c["training"]["loss_weights"]["mb"]=0
            else:c["test_samples"]=1000
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):
                data.validate_settings(c)

    def test_continuous_schedule_is_shared_by_k_and_stable_on_append(self):
        c=settings();jobs=data.jobs_for(c,7)
        for lid in c["training_layout_ids"]:
            cases=[j["cases"] for j in jobs if j["layout_id"]==lid]
            self.assertEqual(cases[0],cases[1])
        self.assertEqual(jobs,data.jobs_for(c,7))
        self.assertTrue(any(case["power_dbm"]!=round(case["power_dbm"],1) for j in jobs for case in j["cases"]))

    def test_label_alpha_is_fixed_by_maximum_declared_groups(self):
        c=settings();jobs=data.jobs_for(c,7)
        p={"config":c,"jobs":jobs,"geometry_spec":{"config":dict(c["label"],profile="geometry"),
            "deployment":{},"array_settings":{}},"pso_settings":{},"plan_sha256":"unit"}
        local=data.label_plan(p,jobs[0])
        self.assertEqual(local["config"]["alpha"]/2,c["label"]["alpha"]/(2*len(jobs)))


class ModelAndFitTests(unittest.TestCase):
    def test_small_models_accept_unseen_k_and_preserve_simplex(self):
        plan,train=fit_plan();bounds=fixture()[0]["bounds"]
        graphs=[rows_for(k,1)[0]["graph"] for k in (2,3,6)]
        configure_runtime(7,1,"cpu")
        for kind in ("gnn","mlp"):
            model=build_model(kind,bounds,plan["config"])
            values=predict(model,graphs,plan["normalization"])
            for graph,y in zip(graphs,values):
                self.assertEqual(len(y),graph["n_public"]+2)
                self.assertAlmostEqual(float(sum(y[:-1])),1.,places=6)
                np.testing.assert_allclose(y,predict(model,[graph],plan["normalization"])[0],atol=2e-7,rtol=0)

    def test_both_model_kinds_resume_exactly(self):
        plan,rows=fit_plan();bounds=fixture()[0]["bounds"]
        for kind in ("gnn","mlp"):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as full,tempfile.TemporaryDirectory() as resumed:
                fit(full,kind,7,plan,bounds,rows,rows)
                with self.assertRaises(InterruptedError):
                    fit(resumed,kind,7,plan,bounds,rows,rows,stop_after_epoch=2)
                fit(resumed,kind,7,plan,bounds,rows,rows)
                a,sa=load_model(Path(full)/f"fits/{kind}_s7/best_weighted.pt")
                b,sb=load_model(Path(resumed)/f"fits/{kind}_s7/best_weighted.pt")
                self.assertEqual(sa["epoch"],sb["epoch"])
                self.assertTrue(all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict()))
                self.assertEqual((Path(full)/f"fits/{kind}_s7/history.csv").read_bytes(),
                                 (Path(resumed)/f"fits/{kind}_s7/history.csv").read_bytes())

    def test_wrong_train_prefix_checkpoint_is_rejected(self):
        plan,rows=fit_plan();bounds=fixture()[0]["bounds"]
        with tempfile.TemporaryDirectory() as folder:
            fit(folder,"gnn",7,plan,bounds,rows,rows)
            altered=dict(plan,training_layout_ids=[2001])
            with self.assertRaisesRegex(ValueError,"metadata"):
                read_fit(folder,"gnn",7,altered)

    def test_audit_field_is_not_an_input_to_model(self):
        plan,train=fit_plan()
        altered=deepcopy(train)
        for row in altered:row["audit"]={"passed":False}
        configure_runtime(7,1,"cpu")
        model=build_model("gnn",fixture()[0]["bounds"],plan["config"])
        a=predict(model,[r["graph"] for r in train],plan["normalization"])
        b=predict(model,[r["graph"] for r in altered],plan["normalization"])
        for first,second in zip(a,b):
            np.testing.assert_array_equal(first,second)

    def test_prediction_records_serialize_variable_k_arrays(self):
        rows=rows_for(2)
        job={"id":"train_l002000_k2","layout_id":2000,"n_public":2,"split":"validation","cases":[{}]*len(rows)}
        stage={"validation_jobs":[job],"normalization":{}}
        refs={"unit":{"kind":"gnn","candidate_id":"unit","path":"checkpoint.pt"}}
        values=[np.array([.2,.4,.4,.5]) for _ in rows]
        with tempfile.TemporaryDirectory() as folder:
            write_json(Path(folder)/"plan.json",{})
            with patch("ris_noma.curve_learning.load_model",return_value=(object(),{})), \
                 patch("ris_noma.curve_learning.predict",return_value=values):
                freeze_predictions(folder,"validation",stage,rows,refs,"frozen")
            record=read_json(Path(folder)/"validation_predictions"/(job["id"]+".json"))
            self.assertEqual(record["rows"][0]["allocation"],[.2,.4,.4,.5])


class SelectionAndPhysicsTests(unittest.TestCase):
    def test_validation_feasibility_precedes_low_loss(self):
        refs={};rows=[]
        for architecture in ("reference","small"):
            for kind in ("gnn","mlp"):
                for tag,loss,passes in (("low_loss",.001,1),("feasible",.01,2)):
                    key=f"{architecture}::{kind}_{tag}"
                    refs[key]={"candidate_id":key,"architecture_name":architecture,"kind":kind,"seed":7,
                               "epoch":2,"validation_errors":{"weighted_loss":loss}}
                    for i in range(2):
                        rows.append({"candidate_id":key,"phase":"validation","split":"validation","passed":i<passes,
                            "gated_bob_rate":4. if i<passes else 0.,"bob_rate":4.})
        plan={"config":{"training":{"seeds":[7]}}}
        selected=select(refs,rows,plan)
        self.assertTrue(selected["views"]["small::gnn_s7_physical"]["candidate_id"].endswith("feasible"))
        self.assertTrue(selected["views"]["small::gnn_s7_weighted_loss"]["candidate_id"].endswith("low_loss"))
        wrong=deepcopy(rows);wrong[0]["phase"]="test"
        with self.assertRaises(ValueError):select(refs,wrong,plan)

    def test_maximum_candidate_family_does_not_change_with_train_size(self):
        a={"max_validation_views":217,"max_train_probe_views":73,
           "config":{"physics":{"alpha":.05}},"validation_jobs":[{}]*64,"train_probe_jobs":[{}]*32,
           "stage_validation_views":73}
        b=dict(a,stage_validation_views=217)
        job={"cases":[{}]*8}
        self.assertEqual(confidence(a,"validation",job),confidence(b,"validation",job))
        self.assertEqual(confidence(a,"validation",job),(1736,.05/128))

    def test_no_channels_before_fits_predictions_and_selection_freeze(self):
        with tempfile.TemporaryDirectory() as folder,patch("ris_noma.curve_physics.context") as bank:
            root=Path(folder);job={"id":"unit"}
            with self.assertRaisesRegex(ValueError,"Freeze all stage fits"):audit_task((folder,"validation",job))
            write_json(root/"fits_frozen.json",{})
            with self.assertRaisesRegex(ValueError,"Freeze every prediction"):audit_task((folder,"validation",job))
            with self.assertRaisesRegex(ValueError,"Select checkpoints"):audit_task((folder,"train_probe",job))
            bank.assert_not_called()

    def test_final_test_role_is_rejected_before_channel_generation(self):
        with tempfile.TemporaryDirectory() as folder,patch("ris_noma.curve_physics.build_bank") as bank:
            with self.assertRaises(ValueError):audit_task((folder,"test",{}))
            with self.assertRaises(ValueError):context({},folder,{},"test")
            with self.assertRaises(ValueError):confidence({},"test",{})
            bank.assert_not_called()

    def test_missing_teacher_label_stays_in_physical_denominator(self):
        source=rows_for(2,2);source[1].update(has_label=False,allocation=None)
        rows=[{"case_id":r["case_id"],"allocation":[.2,.4,.4,.5],"returned":True,
               "passed":i==0,"bob_rate":4.,"gated_bob_rate":4. if i==0 else 0.} for i,r in enumerate(source)]
        score=physical_score(rows)
        errors=allocation_errors(rows,{r["case_id"]:r for r in source},{"mb":32.,"public_total":2.,"s":1.})
        self.assertEqual(score["cases"],2);self.assertEqual(score["pass_fraction"],.5)
        self.assertEqual(errors["labeled_error_cases"],1)

    def test_geometry_rule_failure_is_reported_without_filtering_denominator(self):
        rows=[{"passed":True,"bob_rate":4.,"gated_bob_rate":4.,
               "point_checks_without_dep":{"layout_bob_mean_gain_rule":True}},
              {"passed":False,"bob_rate":4.,"gated_bob_rate":0.,
               "point_checks_without_dep":{"layout_bob_mean_gain_rule":False}}]
        result=physical_score(rows)
        self.assertEqual(result["cases"],2)
        self.assertEqual(result["pass_fraction"],.5)
        self.assertEqual(result["layout_gain_rule_passed_cases"],1)
        self.assertEqual(result["pass_fraction_on_layout_gain_rule_passed"],1.)

    def test_nonempty_or_changed_plan_requires_new_output(self):
        with tempfile.TemporaryDirectory() as folder:
            write_json(Path(folder)/"plan.json",{"workers":2})
            with patch("ris_noma.curve_learning.make_master",return_value={"workers":4}):
                with self.assertRaisesRegex(ValueError,"nonempty"):initialize(folder,"/another/corpus",4,False)
                with self.assertRaisesRegex(ValueError,"Changed"):initialize(folder,"/another/corpus",4,True)

    def test_seal_requires_two_completed_prefixes(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,"at least two"):seal("corpus",folder,"unit")


class LayoutAnalysisTests(unittest.TestCase):
    def test_mixed_teacher_model_and_contrast_columns_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"mixed.csv"
            table(path,[{"kind":"teacher","cases":2},{"kind":"gnn","cases":2,"case_mse":.01}])
            import csv
            rows=list(csv.DictReader(path.open(encoding="utf-8-sig")))
            self.assertEqual(rows[1]["case_mse"],"0.01")
            self.assertEqual(rows[0]["case_mse"],"")

    def test_identical_methods_have_zero_paired_difference_by_layout(self):
        a={i:{"pass_fraction":.8,"gated_rate":4.+i,"case_mse":.01} for i in (1,2,3)}
        cfg=settings()["curve"]
        actual=paired_bootstrap(a,deepcopy(a),cfg,[1])
        self.assertEqual(actual["layout_clusters"],3)
        for metric in ("pass_fraction","gated_rate","case_mse"):
            for suffix in ("difference","ci_low","ci_high"):
                self.assertEqual(actual[metric+"_"+suffix],0.)

    def test_missing_paired_layout_is_an_error(self):
        a={1:{"pass_fraction":1.,"gated_rate":4.,"case_mse":.01}}
        with self.assertRaises(ValueError):paired_bootstrap(a,{},settings()["curve"],[1])

    def test_label_missing_layout_remains_in_physics_but_not_mse_bootstrap(self):
        a={1:{"pass_fraction":1.,"gated_rate":4.,"case_mse":.01},
           2:{"pass_fraction":0.,"gated_rate":0.,"case_mse":None}}
        result=paired_bootstrap(a,deepcopy(a),settings()["curve"],[2])
        self.assertEqual(result["pass_fraction_layout_clusters"],2)
        self.assertEqual(result["case_mse_layout_clusters"],1)
        self.assertEqual(result["case_mse_difference"],0.)


if __name__=="__main__":
    unittest.main()
