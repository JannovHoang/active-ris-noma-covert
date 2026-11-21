"""Benchmark contracts, using a small synthetic pipeline fixture.

These small settings test implementation, not reported project performance.
The delivered experiment uses the user's actual uploaded checkpoints instead.
"""
from copy import deepcopy
from pathlib import Path
import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
try:
    import torch
    from ris_noma.benchmark_pso import (warm_positions,role_rng,settings_for,validate_config_5c,
        select_checkpoints,predict_start,run_benchmark,read_json,atomic_json,sha256,file_fingerprints,load_record)
    from ris_noma.learning_train import configure_runtime,predict,regression_metrics
    from ris_noma.learning_models import make_model
except ModuleNotFoundError as error:
    if error.name!='torch':raise
    torch=None

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipIf(torch is None,'Install requirements_step5c.txt')
class TestWarmStart(unittest.TestCase):
    def test_warm_particles_keep_exact_center_and_normalized_bounds(self):
        center=np.array([0.,.5,1.])
        result=warm_positions(center,6,.4,np.random.default_rng(7))
        np.testing.assert_array_equal(result[0],center)
        self.assertTrue(np.all((result>=0)&(result<=1)))
        self.assertEqual(result.shape,(6,3))

    def test_jitter_does_not_advance_the_optimizer_random_stream(self):
        a=role_rng(7,102,'optimizer',101,0)
        expected=a.uniform(size=(30,3))
        warm_positions(np.ones(3)*.5,6,.05,role_rng(7,102,'warm_jitter',101,0))
        b=role_rng(7,102,'optimizer',101,0)
        np.testing.assert_array_equal(expected,b.uniform(size=(30,3)))

    def test_only_replaced_particles_differ_from_paired_random_initialization(self):
        from ris_noma.pso import optimize,PSOSettings,Outcome
        settings=PSOSettings(swarm_size=6,sweeps=1)
        positions=[]
        def f(z):positions.append(z);return Outcome(float(z.sum()),0.,True)
        optimize(f,3,settings,role_rng(7,102,'optimizer',101))
        random=np.array(positions);positions.clear()
        initial=warm_positions(np.ones(3)*.5,2,.05,role_rng(7,102,'warm_jitter',101))
        optimize(f,3,settings,role_rng(7,102,'optimizer',101),initial)
        warm=np.array(positions)
        np.testing.assert_array_equal(warm[:2],initial)
        np.testing.assert_array_equal(warm[2:],random[2:])

    def test_audit_stream_is_distinct_from_previously_used_heldout_banks(self):
        from ris_noma.validation import fresh_rng
        new=role_rng(7,102,'audit',1).normal(size=30)
        for seed in [1,2,3]:self.assertFalse(np.array_equal(new,fresh_rng(7,102,'heldout',seed).normal(size=30)))

    def test_config_keeps_random_particles_and_a_smaller_short_budget(self):
        c=read_json(ROOT/'configs/benchmark_pilot_5c.json')
        for key,value in [('warm_particles',30),('short_sweeps',60),('pso_seeds',[1,1]),('audit_samples',1)]:
            bad=deepcopy(c);bad[key]=value
            with self.assertRaises(ValueError):validate_config_5c(bad)


@unittest.skipIf(torch is None,'Install requirements_step5c.txt')
class TestBenchmarkPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import run_step3b,run_step5a
        from ris_noma.config import load_config
        from ris_noma.pso import PSOSettings
        from ris_noma.learning_data import load_split,TARGET_KEY
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        c=load_config(ROOT/'configs/legacy_physics_k2_pilot.json')
        c['requirements'].update(xi=[.2,.4],qos_public=[.01,.01],target_public=[.01,.01],target_bob=.01)
        policy=run_step3b.load_policy(ROOT/'configs/validation_pilot.json')
        policy.update(label_samples=150,validation_samples=1500,heldout_samples=1500,margins=[0.,.02])
        partition={'purpose':'pilot','splits':{'train':[100],'validation':[101],'test':[102]},'pso_seeds':[1],'study_alpha':.05,'heldout_seed_new':2}
        with contextlib.redirect_stdout(io.StringIO()):
            run_step5a.generate(c,PSOSettings(swarm_size=4,sweeps=2),policy,partition,cls.root/'dataset')
        cls.dataset=cls.root/'dataset/graphs';cls.training=cls.root/'training';cls.training.mkdir()
        cls.normalizer=read_json(cls.dataset/'normalization.json');cls.schema=read_json(cls.dataset/'feature_schema.json')
        cls.bounds=read_json(cls.dataset/'dataset_manifest.json')['bounds']
        tc=read_json(ROOT/'configs/training_pilot_5b.json');tc.update(gnn_hidden=8,gnn_layers=1,mlp_hidden=8,mlp_layers=1)
        val=load_split(cls.dataset,'validation');fingerprint=file_fingerprints(cls.dataset)
        rows=[];hashes={}
        for kind in ['gnn','mlp']:
            for seed in [7,17]:
                configure_runtime(seed);model=make_model(kind,cls.bounds,tc)
                mse=regression_metrics(predict(model,val),val[TARGET_KEY])['mse']
                relative=f'{kind}_seed_{seed}/best_model.pt';path=cls.training/relative;path.parent.mkdir()
                torch.save({'format':'ris_noma_step5b_v1','model_kind':kind,'seed':seed,'config':tc,'bounds':cls.bounds,
                    'feature_schema':cls.schema,'normalization':cls.normalizer,'dataset_sha256':fingerprint,'best_epoch':0,
                    'selection_metric':'validation_allocation_mse','test_used_for_selection':False,'state_dict':model.state_dict()},path)
                hashes[relative]=sha256(path)
                rows.append({'model':kind,'seed':seed,'best_epoch':0,'validation':{'mse':mse},'test':{'mse':1000.-mse},'test_used_for_selection':False})
        atomic_json(cls.training/'training_status.json',{'training_complete':True,'all_declared_seeds_reported':True,'runs':rows,'checkpoint_sha256':hashes})
        atomic_json(cls.training/'training_plan.json',{'dataset_sha256':fingerprint,'test_used_for_training_or_checkpoint_selection':False})
        cls.config=read_json(ROOT/'configs/benchmark_pilot_5c.json')
        cls.config.update(splits=['test'],pso_seeds=[101],swarm_size=4,full_sweeps=3,short_sweeps=2,warm_particles=1,audit_samples=1000)
        cls.output=cls.root/'result'
        import ris_noma.benchmark_pso as benchmark
        original=benchmark.audit_one
        def checking_audit(*args,**kwargs):
            frozen=read_json(cls.output/'search_frozen.json')
            assert frozen['count']==8
            assert len(frozen['search_records_sha256'])==8
            for key,digest in frozen['search_records_sha256'].items():assert sha256(cls.output/'search'/key)==digest
            return original(*args,**kwargs)
        with contextlib.redirect_stdout(io.StringIO()),patch.object(benchmark,'audit_one',side_effect=checking_audit):
            cls.result=run_benchmark(cls.dataset,cls.training,cls.output,cls.config)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_validation_only_selection_and_all_checkpoint_hashes_checked(self):
        selected,_,_=select_checkpoints(self.training,self.dataset,self.config)
        status=read_json(self.training/'training_status.json')
        for kind in ['gnn','mlp']:
            expected=min([r for r in status['runs'] if r['model']==kind],key=lambda r:r['validation']['mse'])
            self.assertEqual(selected[kind]['training_seed'],expected['seed'])
            self.assertFalse(selected[kind]['test_used_for_choice'])
        altered=self.root/'corrupt_model';shutil.copytree(self.training,altered)
        target=altered/selected['gnn']['path']
        with target.open('ab') as f:f.write(b'changed')
        with self.assertRaisesRegex(ValueError,'checkpoint changed'):select_checkpoints(altered,self.dataset,self.config)

    def test_prediction_start_mapping_is_correct_and_does_not_read_label_fields(self):
        from ris_noma.learning_data import load_split
        from ris_noma.objective import decode_k2
        _,models,_=select_checkpoints(self.training,self.dataset,self.config)
        d=load_split(self.dataset,'test');i=0
        graph={k:d[k][i].copy() for k in ['node_raw','global_raw','edge_index','edge_attr']}
        order=d['sic_order_original_ids'][i]
        y,z=predict_start(*models['gnn'],graph,order,self.bounds)
        graph['allocation']=np.full(4,10000.);graph['reference_rate']=1e100
        y2,z2=predict_start(*models['gnn'],graph,order,self.bounds)
        np.testing.assert_array_equal(y,y2);np.testing.assert_array_equal(z,z2)
        mb,mp,s=decode_k2(z,self.bounds)
        np.testing.assert_allclose([mb,*mp,s],[y[0],*y[1:3][order],y[3]],atol=1e-12)

    def test_complete_pipeline_has_budget_and_all_audits_without_filtering(self):
        self.assertTrue(self.result['benchmark_complete']);self.assertEqual(self.result['search_runs'],8)
        self.assertEqual(self.result['total_pso_evaluations'],72)
        self.assertFalse(self.result['heldout_used_for_search_or_reselection'])
        frozen=read_json(self.output/'search_frozen.json')
        for key in frozen['search_records_sha256']:
            search=load_record(self.output/'search'/key);audit=load_record(self.output/'audit'/key)
            self.assertFalse(search['heldout_used_for_search']);self.assertFalse(search['source_label_allocation_used_for_start'])
            self.assertAlmostEqual(audit['total_decision_seconds'],audit['search_wall_seconds']+audit['audit_wall_seconds'])
            self.assertEqual(audit['audit_gated_bob_rate'],audit['bob_rate'] if audit['bound_pass'] else 0.)
            self.assertEqual(audit['screen']['constants']['pool_size'],8)

    def test_resume_skips_finished_searches_and_audits(self):
        with contextlib.redirect_stdout(io.StringIO()),patch('ris_noma.benchmark_pso.search_one',side_effect=AssertionError('search must be skipped')),patch('ris_noma.benchmark_pso.audit_one',side_effect=AssertionError('audit must be skipped')):
            result=run_benchmark(self.dataset,self.training,self.output,self.config,resume=True)
        self.assertEqual(result,self.result)

    def test_changed_plan_is_rejected_on_resume(self):
        changed=deepcopy(self.config);changed['search_dep_margin']=.03
        with self.assertRaisesRegex(ValueError,'Nonempty/changed'):
            run_benchmark(self.dataset,self.training,self.output,changed,resume=True)

    def test_tampered_search_is_rejected_before_using_its_audit(self):
        destination=self.root/'corrupt_result';shutil.copytree(self.output,destination)
        key=next(iter(read_json(destination/'search_frozen.json')['search_records_sha256']))
        with (destination/'search'/key).open('ab') as f:f.write(b'changed')
        with self.assertRaisesRegex(ValueError,'Changed or incomplete'):
            run_benchmark(self.dataset,self.training,destination,self.config,resume=True)

    def test_plot_process_works_without_importing_torch(self):
        command="import sys; from ris_noma.benchmark_plots import plot_benchmark; assert 'torch' not in sys.modules; plot_benchmark(sys.argv[1]); assert 'torch' not in sys.modules"
        subprocess.run([sys.executable,'-c',command,str(self.output)],cwd=ROOT,check=True,capture_output=True)
        for name in ['overview.png','rate_and_feasibility.png','time_and_speedup.png','convergence.png','margins_and_gated_rate.png']:
            self.assertGreater((self.output/'figures'/name).stat().st_size,1000)


if __name__=='__main__':unittest.main()
