"""7c protocol/model tests; synthetic labels are isolated unit-test fixtures."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ris_noma.config import load_config
from ris_noma.channels import public_order
from ris_noma.learning_data import read_json, sha256
from ris_noma.learning_train import configure_runtime
from ris_noma.multik_data_core import context as data_context
from ris_noma.multik_dataset import make_jobs, write_json, write_jsonl, read_jsonl
from ris_noma.multik_graph import fit_normalizer, pack_graphs, statistical_graph
from ris_noma.multik_models import tensor_batch
from ris_noma.multik_learning_models import build_model, error_metrics, load_model
from ris_noma.multik_learning_fit import fit
from ris_noma.multik_learning_physics import audit_task
from ris_noma import multik_learning as learning

ROOT=Path(__file__).resolve().parents[1]


def fixture(folder):
    dataset=Path(folder)/"dataset";dataset.mkdir()
    source={"base_config":load_config(ROOT/"configs/legacy_physics_k2_pilot.json"),
            "experiment":{"position_half_width_m":{"bob":5.,"willie":4.}}}
    dc=read_json(ROOT/"configs/multik_dataset_7b.json")
    dc.update(training_public_counts=[2,3],unseen_public_counts=[4],
              splits={"train":[1300],"validation":[1301],"test":[1302]},power_dbm=[35.],xi=[.75,.95],
              search_samples=1000,selection_samples=1000,heldout_samples=1000)
    from ris_noma.multik_data_core import STREAMS as DATA_STREAMS
    dp={"config":dc,"streams":DATA_STREAMS,"jobs":make_jobs(dc),"code_sha256":{}}
    rows={role:[] for role in ("train","validation","test","test_unseen_k")}
    for job in dp["jobs"]:
        physical,layout,bank=data_context(dp,source,job,"search")
        order=public_order(bank)
        for xi in dc["xi"]:
            graph=statistical_graph(physical,layout,bank,order,xi)
            # A fixed valid simplex target tests plumbing; it is NOT a claim
            # of PSO success or a scientific dataset supplied to the user.
            y=[.12]+[.88/job["n_public"]]*job["n_public"]+[.5]
            row={**job,"power_dbm":35.,"xi":xi,"graph":graph,"allocation":y,"order":order.tolist(),
                 "label_available":True,"heldout_used_for_label_filter":False,"heldout_passed":True}
            role="test_unseen_k" if job["regime"]=="unseen_k" else job["split"]
            rows[role].append(row)
    for role,values in rows.items():write_jsonl(dataset/"graphs"/f"{role}.jsonl",values)
    for name,value in (("plan.json",dp),("source_physics_plan.json",source),("completion.json",{"unit_test_fixture":True}),("feature_schema.json",{"unit_test_fixture":True})):
        write_json(dataset/name,value)
    status={"counts":{role:{"labels":len(values)} for role,values in rows.items()}}
    c=read_json(ROOT/"configs/multik_training_smoke_7c.json")
    c.update(epochs=3,snapshot_epochs=[1,3],batch_size=4,evaluation_batch_size=16,hidden=8,layers=1,mlp_inner=28,
             validation_physics_samples=1000,test_physics_samples=1000)
    return dataset,source,dp,status,c


def hashes(folder):
    return {str(p.relative_to(folder)):sha256(p) for p in Path(folder).rglob("*") if p.is_file()}


class LearningModelTests(unittest.TestCase):
    def setUp(self):
        configure_runtime(7,1,"cpu")
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.dataset,self.source,self.dp,self.status,self.config=fixture(self.temp.name)
        self.rows=learning.load_role(self.dataset,"train",self.dp)
        self.norm=fit_normalizer([r["graph"] for r in self.rows])

    def test_parameter_match_and_same_model_accepts_unseen_k(self):
        c=read_json(ROOT/"configs/multik_training_7c.json")
        gnn=build_model("gnn",self.source["base_config"]["bounds"],c)
        mlp=build_model("mlp",self.source["base_config"]["bounds"],c)
        self.assertEqual(sum(p.numel() for p in gnn.parameters()),122883)
        self.assertEqual(sum(p.numel() for p in mlp.parameters()),122595)
        for kind in ("gnn","mlp"):
            model=build_model(kind,self.source["base_config"]["bounds"],self.config)
            for role in ("train","test_unseen_k"):
                rows=learning.load_role(self.dataset,role,self.dp)
                batch=tensor_batch(pack_graphs([r["graph"] for r in rows],self.norm))
                y=model(**batch)
                self.assertTrue(torch.isfinite(y).all())
                self.assertTrue(torch.allclose(y[:,:-1].sum(-1),torch.ones(len(rows)),atol=1e-6))
                self.assertEqual(y.shape[1],max(r["n_public"] for r in rows)+2)

    def test_pooled_mlp_does_not_use_edges_and_ignores_padded_nodes(self):
        model=build_model("mlp",self.source["base_config"]["bounds"],self.config)
        graphs=[r["graph"] for r in self.rows]
        batch=tensor_batch(pack_graphs(graphs,self.norm))
        first=model(**batch)
        altered={key:value.clone() for key,value in batch.items()}
        altered["edge_attr"][:]=100.
        altered["node"][~altered["node_mask"]]=1000.
        second=model(**altered)
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        for i,graph in enumerate(graphs):
            single=model(**tensor_batch(pack_graphs([graph],self.norm)))[0]
            actual=torch.cat([first[i,:graph["n_public"]+1],first[i,-1:]])
            torch.testing.assert_close(single,actual,rtol=1e-5,atol=1e-7)

    def test_public_permutation_equivariance_both_models(self):
        g=deepcopy(self.rows[0]["graph"])
        permutation=np.array([0,2,1]);inverse=np.argsort(permutation)
        swapped=deepcopy(g)
        swapped["node_raw"]=np.array(g["node_raw"])[permutation].tolist()
        swapped["edge_index"]=inverse[np.array(g["edge_index"])].tolist()
        for kind in ("gnn","mlp"):
            model=build_model(kind,self.source["base_config"]["bounds"],self.config)
            a=model(**tensor_batch(pack_graphs([g],self.norm)))[0]
            b=model(**tensor_batch(pack_graphs([swapped],self.norm)))[0]
            torch.testing.assert_close(b,torch.cat([a[permutation],a[-1:]]),rtol=1e-5,atol=1e-7)

    def test_allocation_errors_ignore_padding_and_balance_k(self):
        targets=torch.zeros((2,5));prediction=torch.ones((2,5))*.1
        masks=torch.tensor([[True,True,False,False],[True,True,True,True]])
        first=error_metrics(prediction,targets,masks,self.config["loss_weights"])
        prediction[0,2:4]=100.
        second=error_metrics(prediction,targets,masks,self.config["loss_weights"])
        self.assertEqual(first,second)
        self.assertAlmostEqual(first["case_mse"],.01,places=7)
        self.assertAlmostEqual(first["weighted_loss"],.01,places=7)


class LearningPolicyTests(unittest.TestCase):
    def test_invalid_training_config_rejected(self):
        c=read_json(ROOT/"configs/multik_training_7c.json")
        for changes in ({"seeds":[7,7]},{"snapshot_epochs":[20,60]},{"snapshot_epochs":[20,121]},
                        {"alpha":1.},{"learning_rate":float("nan")},{"resume_every":0},
                        {"loss_weights":{"mb":32.,"public_total":-2.,"s":1.}}, {"test_physics_samples":999}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):learning.validate_config(c|changes)

    def test_physics_selection_differs_from_lowest_loss_and_rejects_test(self):
        refs={};rows=[]
        for kind in ("gnn","mlp"):
            for index in (1,2):
                key=f"{kind}{index}"
                refs[key]={"kind":kind,"seed":7,"epoch":index,"path":"fixture.pt","validation_errors":{"weighted_loss":index*.01}}
                for j in range(2):
                    rows.append({"candidate_id":key,"split":"validation","regime":"seen_k",
                                 "passed":index==2,"bob_rate":4.,"gated_bob_rate":4. if index==2 else 0.})
        plan={"config":{"seeds":[7]},"labeled_counts":{"validation":2},"physical_selection":"fixture"}
        selected=learning.select_checkpoints(refs,rows,plan)
        self.assertEqual(selected["views"]["gnn_s7_weighted_loss"]["epoch"],1)
        self.assertEqual(selected["views"]["gnn_s7_physical"]["epoch"],2)
        rows[0]["split"]="test"
        with self.assertRaises(ValueError):learning.select_checkpoints(refs,rows,plan)

    def test_model_test_bank_requires_frozen_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_json(Path(tmp)/"plan.json",{})
            with self.assertRaises(ValueError):audit_task((tmp,"test",{"layout_id":1302,"n_public":2,"split":"test","regime":"seen_k"}))


class LearningRunTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
        self.dataset,self.source,self.dp,self.source_status,self.config=fixture(self.folder)
        self.output=self.folder/"result"

    def inspect(self,config=None):
        with patch("ris_noma.multik_learning.check_dataset",return_value=self.source_status):
            return learning.inspect(self.dataset,self.config if config is None else config)

    def generate(self,output=None,workers=1,resume=False,config=None):
        with patch("ris_noma.multik_learning.check_dataset",return_value=self.source_status):
            return learning.run(self.dataset,self.output if output is None else output,self.config if config is None else config,workers,resume)

    def test_inspection_normalizes_train_only_and_loader_ignores_audit_flag(self):
        plan,*_=self.inspect()
        train=learning.load_role(self.dataset,"train",self.dp)
        original=read_jsonl(self.dataset/"graphs/train.jsonl")
        changed=deepcopy(original)
        for r in changed:r["heldout_passed"]=False
        write_jsonl(self.dataset/"graphs/train.jsonl",changed)
        self.assertEqual(train,learning.load_role(self.dataset,"train",self.dp))
        val=read_jsonl(self.dataset/"graphs/validation.jsonl")
        for r in val:r["graph"]["node_raw"][0][2]+=100.
        write_jsonl(self.dataset/"graphs/validation.jsonl",val)
        updated,*_=self.inspect()
        self.assertEqual(plan["normalization"],updated["normalization"])

    def test_optimizer_resume_reproduces_uninterrupted_weights_and_history(self):
        c=self.config|{"epochs":5,"snapshot_epochs":[2,5]}
        plan,source,_,train,val=self.inspect(c)
        active=dict(plan,plan_sha256="unit_test_fixed_plan")
        first=self.folder/"continuous";second=self.folder/"resumed"
        fit(first,"gnn",7,active,source["base_config"]["bounds"],train,val)
        with self.assertRaises(InterruptedError):fit(second,"gnn",7,active,source["base_config"]["bounds"],train,val,stop_after_epoch=2)
        fit(second,"gnn",7,active,source["base_config"]["bounds"],train,val)
        for relative in ("best_weighted.pt","snapshots/epoch_0002.pt","snapshots/epoch_0005.pt"):
            a=torch.load(first/"fits/gnn_s7"/relative,map_location="cpu",weights_only=True)
            b=torch.load(second/"fits/gnn_s7"/relative,map_location="cpu",weights_only=True)
            for key in a["state_dict"]:torch.testing.assert_close(a["state_dict"][key],b["state_dict"][key],rtol=0,atol=0)
        self.assertEqual((first/"fits/gnn_s7/history.csv").read_bytes(),(second/"fits/gnn_s7/history.csv").read_bytes())

    def test_corrupt_optimizer_resume_is_rejected(self):
        plan,source,_,train,val=self.inspect()
        active=dict(plan,plan_sha256="unit_test_fixed_plan")
        with self.assertRaises(InterruptedError):fit(self.output,"mlp",7,active,source["base_config"]["bounds"],train,val,stop_after_epoch=1)
        pointer=self.output/"fits/mlp_s7/resume.json"
        saved=read_json(pointer);saved["sha256"]="changed";write_json(pointer,saved)
        with self.assertRaises(ValueError):fit(self.output,"mlp",7,active,source["base_config"]["bounds"],train,val)

    def test_complete_protocol_and_source_preservation(self):
        before=hashes(self.dataset)
        result=self.generate()
        self.assertTrue(result["complete"])
        self.assertEqual(result["fits"],2)
        self.assertEqual(result["candidate_checkpoints"],6)
        self.assertEqual(result["selected_views"],4)
        self.assertEqual(result["validation_rows"],24)
        self.assertEqual(result["test_rows"],30)
        self.assertEqual(before,hashes(self.dataset))
        self.assertTrue(learning.check(self.output,self.dataset)["complete"])
        selected=read_json(self.output/"selection_frozen.json")
        self.assertFalse(selected["test_used_for_selection"])
        self.assertEqual(len(result["metrics"]["by_view"]),5)

    def test_existing_output_and_changed_resume_are_rejected_without_mutation(self):
        self.generate();before=hashes(self.output)
        with self.assertRaises(ValueError):self.generate()
        with self.assertRaises(ValueError):self.generate(resume=True,config=self.config|{"learning_rate":.002})
        self.assertEqual(before,hashes(self.output))
        self.assertTrue(self.generate(resume=True)["complete"])
        self.assertEqual(before,hashes(self.output))

    def test_parallel_physics_matches_sequential_rows(self):
        self.generate()
        other=self.folder/"parallel"
        self.generate(output=other,workers=2)
        for phase in ("validation","test"):
            for path in (self.output/f"{phase}_audit").glob("layout_*.json"):
                if path.name.endswith(".receipt.json"):continue
                self.assertEqual(read_json(path)["rows"],read_json(other/f"{phase}_audit"/path.name)["rows"])


if __name__=="__main__":unittest.main()
