from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase,main
from unittest.mock import patch
import copy
import json
import numpy as np
import torch

import step7g_extra
from step7g_extra import baseline,analysis,physics
from step7g_extra.common import read,write,digest,commit,committed,settings,allocation_errors
from ris_noma.multik_learning_models import error_metrics

BOUNDS={"mb":[.05,.49],"s":[.05,.95]}


def fixture():
    rows=[]
    for i in range(24):
        k=2+i%4;mb=.15+.01*(i%5);public=np.exp(-.2*np.arange(k));public*= (1-mb)/public.sum()
        rows.append({"case_id":str(i),"split":"train","has_label":True,"layout_id":3000+i//8,
            "n_public":k,"power_dbm":30+10*i/24,"xi":.75+.2*(i%8)/8,
            "order":list(range(k)),"allocation":[mb,*public,.4+.02*(i%5)],"graph":{"fake_audit":i}})
    return rows


class BaselineTests(TestCase):
    def test_no_geometry_or_audit_shortcut(self):
        rows=fixture();model=baseline.fit(rows,BOUNDS,1.)
        query=copy.deepcopy(rows)
        for r in query:r["graph"]={"leak":123};r["allocation"]=[123];r["has_label"]=False;r["layout_id"]=999999;r["audit"]="bad"
        for a,b in zip(baseline.predict(model,rows),baseline.predict(model,query)):np.testing.assert_array_equal(a,b)

    def test_validation_cannot_fit_baseline(self):
        rows=fixture();rows[0]["split"]="validation"
        with self.assertRaises(ValueError):baseline.fit(rows,BOUNDS,1.)

    def test_sic_order_permutation_is_equivariant(self):
        rows=fixture();model=baseline.fit(rows,BOUNDS,.01);r=rows[3];q=copy.deepcopy(r)
        q["order"]=list(reversed(r["order"]))
        a=baseline.predict(model,[r])[0];b=baseline.predict(model,[q])[0]
        np.testing.assert_allclose(a[1:-1],b[1:-1][q["order"]],atol=1e-12)
        self.assertAlmostEqual(a[:-1].sum(),1.)

    def test_delta_preserves_simplex_ratios_and_s(self):
        a=np.array([.30,.45,.25,.6]);b=baseline.lower_bob(a,.005,BOUNDS)
        self.assertAlmostEqual(b[0],.295);self.assertAlmostEqual(b[:-1].sum(),1.)
        self.assertAlmostEqual(b[1]/b[2],a[1]/a[2]);self.assertEqual(a[-1],b[-1]);self.assertGreater(b[1],a[1])

    def test_delta_respects_mb_floor(self):
        b=baseline.lower_bob([.051,.5,.449,.6],.005,BOUNDS)
        self.assertEqual(b[0],.05);self.assertAlmostEqual(b[:-1].sum(),1.)
        with self.assertRaises(ValueError):baseline.lower_bob(b,-.1,BOUNDS)


class EvidenceTests(TestCase):
    def test_receipt_detects_tampering(self):
        with TemporaryDirectory() as d:
            p=Path(d)/"record.json";commit(p,{"a":1});self.assertEqual(committed(p),{"a":1})
            write(p,{"a":2})
            with self.assertRaises(ValueError):committed(p)

    def test_uncommitted_interruption_can_recompute(self):
        with TemporaryDirectory() as d:
            p=Path(d)/"record.json";write(p,{"partial":1});self.assertIsNone(committed(p))

    def test_allocation_metrics_match_original_torch_definition(self):
        weights={"mb":32.,"public_total":2.,"s":1.}
        for k in (2,3,5):
            target=np.r_[.2,np.full(k,.8/k),.5];prediction=baseline.lower_bob(target,.003,BOUNDS);prediction[-1]+=.02
            ours=allocation_errors(prediction,target,weights)
            expected=error_metrics(torch.tensor(prediction[None,:]),torch.tensor(target[None,:]),torch.ones((1,k+1),dtype=torch.bool),weights)
            self.assertAlmostEqual(ours["case_mse"],expected["case_mse"],places=13)
            self.assertAlmostEqual(ours["weighted_loss"],expected["weighted_loss"],places=13)

    def test_main_study_cannot_silently_reduce_sample_count(self):
        config=read(Path(__file__).resolve().parents[1]/"configs/diagnostics_7g32.json")
        settings(config);config["physics_samples"]=1000
        with self.assertRaises(ValueError):settings(config)

    def test_bank_generation_waits_for_prediction_barrier(self):
        with TemporaryDirectory() as d:
            write(Path(d)/"plan.json",{})
            with patch.object(physics,"context",side_effect=AssertionError("must not generate bank")):
                with self.assertRaises(ValueError):physics.audit_task((d,"validation",{"id":"x"}))

    def test_bootstrap_counts_layout_clusters_not_seed_outputs(self):
        a=[{"layout_id":lid,"passed":True,"gated_bob_rate":3.,"bob_rate":3.,"case_mse":.01,"weighted_loss":.01}
           for lid in (1,2,3) for seed in (7,17,27) for case in range(4)]
        r=analysis.bootstrap_difference(a,a,100,7)
        self.assertEqual(r["layout_clusters"],3)
        for k in ("pass_fraction","gated_rate","case_mse"):
            self.assertEqual(r[k+"_difference"],0.);self.assertEqual(r[k+"_ci_low"],0.);self.assertEqual(r[k+"_ci_high"],0.)

    def test_selection_uses_validation_and_two_stage_baseline_rule(self):
        refs={"frozen":{"family":"frozen32","size":32,"kind":"gnn","seed":7,"epoch":150,"architecture":"small"},
              "a01":{"family":"baseline","size":32,"kind":"baseline","seed":None,"epoch":0,"architecture":"ridge"},
              "a100":{"family":"baseline","size":32,"kind":"baseline","seed":None,"epoch":0,"architecture":"ridge"},
              "teacher":{"family":"teacher","size":32,"kind":"teacher","seed":None,"epoch":0,"architecture":"teacher"}}
        for size in (8,16,32):
            for kind in ("gnn","mlp"):
                refs[f"{size}_{kind}"]={"family":"curve","size":size,"kind":kind,"seed":7,"epoch":20,"architecture":"small"}
        plan={"settings":{"sizes":[8,16,32],"seeds":[7],"deltas":[0.,.005]}}
        views=physics.variants(refs,plan["settings"]);rows=[]
        for k,v in views.items():
            passed=not (v["model_id"]=="a01" and v["delta"]==0)
            rate=5. if v["model_id"]=="a01" else 4.
            rows.append({"view_id":k,"case_id":"x","phase":"validation","split":"validation","passed":passed,
                "gated_bob_rate":rate if passed else 0.,"bob_rate":rate,"case_mse":.01,"weighted_loss":.01})
        with TemporaryDirectory() as d:
            write(Path(d)/"plan.json",plan);write(Path(d)/"validation_audit_frozen.json",{})
            with patch.object(analysis,"read_models",return_value=refs),patch.object(analysis,"phase_rows",return_value=rows):
                selected=analysis.select(d,plan,save=False)
            self.assertTrue(selected["baseline"]["raw"].startswith("a100"))
            self.assertTrue(selected["baseline"]["corrected"].startswith("a100"))
            wrong=copy.deepcopy(rows);wrong[0]["phase"]="train";wrong[0]["split"]="train"
            with patch.object(analysis,"read_models",return_value=refs),patch.object(analysis,"phase_rows",return_value=wrong):
                with self.assertRaises(ValueError):analysis.select(d,plan,save=False)


if __name__=="__main__":main()
