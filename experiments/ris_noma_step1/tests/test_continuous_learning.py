"""7f invariants: no audit/label leakage, variable K, exact resume and bank barriers."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from ris_noma.channels import build_bank,public_order
from ris_noma.config import load_config
from ris_noma.continuous_data import STREAMS as DATA_STREAMS
from ris_noma.continuous_learning import calculate_metrics,run,select_checkpoints
from ris_noma.continuous_learning_data import load_cases,labeled,validate_config,validate_rows
from ris_noma.continuous_learning_fit import fit
from ris_noma.continuous_learning_models import build_model,load_model
from ris_noma.continuous_learning_physics import STREAMS,audit_task,confidence,context
from ris_noma.learning_data import read_json
from ris_noma.learning_train import configure_runtime
from ris_noma.multik_data_core import geometry
from ris_noma.multik_dataset import write_json
from ris_noma.multik_graph import fit_normalizer,pack_graphs,statistical_graph
from ris_noma.multik_models import balanced_allocation_loss,predict

ROOT=Path(__file__).resolve().parents[1]


def fixture(k=2):
    source={'base_config':load_config(ROOT/'configs/legacy_physics_k2_pilot.json'),
            'experiment':{'position_half_width_m':{'bob':5.,'willie':4.}}}
    c,layout=geometry({'config':{'power_dbm':[35.]}},source,{'layout_id':2000,'n_public':k},128)
    bank=build_bank(c,layout,np.random.default_rng(12));order=public_order(bank)
    return c,layout,bank,order,source


def rows_for(k=2,count=3):
    c,layout,bank,order,_=fixture(k);rows=[]
    for i in range(count):
        xi=.8+i*.02;mb=.2+i*.01;power=32.7+i
        rows.append({'case_id':f'train_l002000_k{k}_c{i:03d}','layout_id':2000,'n_public':k,
            'split':'train','regime':'seen_k','role':'train','power_dbm':power,'xi':xi,
            'order':order.tolist(),'graph':statistical_graph(dict(c,total_dbm=power),layout,bank,order,xi),
            'allocation':[mb,*([(1-mb)/k]*k),.5],'has_label':True})
    return rows


def small_plan():
    c=read_json(ROOT/'configs/continuous_training_smoke_7f.json')
    c.update(epochs=4,snapshot_epochs=[2,4],hidden=8,layers=1,mlp_inner=16)
    rows=rows_for(2)+rows_for(3)
    return {'config':c,'normalization':fit_normalizer([r['graph'] for r in rows]),
        'plan_sha256':'unit-fixed-plan','training_public_counts':[2,3],
        'power_dbm_range':[30.,40.],'dep_range':[.75,.95],'profile':'geometry','geometry_scope':{'id':'unit'},
        'dataset_sha256':{'completion.json':'unit-dataset','source_physics_plan.json':'unit-physics'}},rows


class DataBoundaryTests(unittest.TestCase):
    def test_continuous_case_join_retains_missing_labels_and_discards_audit_fields(self):
        source=rows_for();job={'id':'train_l002000_k2','layout_id':2000,'n_public':2,'split':'train','regime':'seen_k',
                              'cases':[{'case_index':i,'power_dbm':r['power_dbm'],'xi':r['xi']} for i,r in enumerate(source)]}
        cases=[{k:r[k] for k in ('case_id','graph','power_dbm','xi')} for r in source]
        label={k:source[0][k] for k in ('case_id','layout_id','n_public','split','regime','power_dbm','xi','graph','allocation')}
        label.update(audit_used_for_label_filter=False,audit={'passed':False})
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);write_json(root/'search'/(job['id']+'.json'),{'cases':cases,'order':source[0]['order']})
            (root/'graphs').mkdir();(root/'graphs/train.jsonl').write_text(__import__('json').dumps(label)+'\n',encoding='utf-8')
            loaded=load_cases(root,'train',{'jobs':[job],'planned_counts':{'train':3}})
        self.assertEqual(len(loaded),3);self.assertEqual(len(labeled(loaded)),1)
        self.assertEqual(loaded[0]['power_dbm'],32.7);self.assertEqual(loaded[0]['xi'],.8)
        self.assertTrue(all('audit' not in r for r in loaded));self.assertIsNone(loaded[1]['allocation'])

    def test_split_k_order_and_audit_leakage_are_rejected(self):
        rows=rows_for();plan={'config':{'splits':{'train':[2000],'validation':[2001],'test':[2002]},
                                       'training_public_counts':[2,3],'unseen_public_counts':[6]}}
        bounds=fixture()[0]['bounds'];validate_rows(rows,'train',plan,bounds)
        for mutate in ('split','layout','k','audit','order'):
            bad=deepcopy(rows)
            if mutate=='split':bad[0]['split']='test'
            elif mutate=='layout':bad[0]['layout_id']=2002
            elif mutate=='k':bad[0]['n_public']=6
            elif mutate=='audit':bad[0]['audit']={'passed':True}
            else:bad[0]['order']=list(reversed(bad[0]['order']))
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):validate_rows(bad,'train',plan,bounds)

    def test_invalid_snapshot_or_duplicate_seed_budget_is_rejected(self):
        c=read_json(ROOT/'configs/continuous_training_7f.json')
        for change in ({'seeds':[7,7]},{'snapshot_epochs':[20,100]},{'validation_physics_samples':999},
                       {'scope':'final_paper'},{'loss_weights':{'mb':0.,'public_total':2.,'s':1.}}):
            with self.subTest(change=change),self.assertRaises(ValueError):validate_config(dict(c,**change))


class ModelAndResumeTests(unittest.TestCase):
    def test_mixed_k_batch_and_k6_preserve_bounds_simplex_and_padding(self):
        configure_runtime(7,1,'cpu');plan,train=small_plan();bounds=fixture()[0]['bounds']
        graphs=[rows_for(k,1)[0]['graph'] for k in (2,3,6)]
        for kind in ('gnn','mlp'):
            model=build_model(kind,bounds,plan['config']);together=predict(model,graphs,plan['normalization'])
            for graph,y in zip(graphs,together):
                alone=predict(model,[graph],plan['normalization'])[0]
                np.testing.assert_allclose(y,alone,atol=2e-7,rtol=0)
                self.assertEqual(len(y),graph['n_public']+2)
                self.assertAlmostEqual(sum(y[:-1]),1.,places=6)
                self.assertTrue(.05<=y[0]<=.49 and .05<=y[-1]<=.95 and np.all(y[1:-1]>=0))

    def test_mlp_output_ignores_edges_and_public_node_permutation_is_equivariant(self):
        configure_runtime(7,1,'cpu');plan,_=small_plan();g=rows_for(3,1)[0]['graph'];bounds=fixture()[0]['bounds']
        altered=deepcopy(g);altered['edge_attr']=(1-np.asarray(g['edge_attr'])).tolist()
        # Packing validates SIC, so edge-insensitivity is tested on tensors.
        model=build_model('mlp',bounds,plan['config'])
        from ris_noma.multik_models import tensor_batch
        tensors=tensor_batch(pack_graphs([g],plan['normalization']))
        with torch.inference_mode():
            original=model(**tensors);tensors['edge_attr']=1-tensors['edge_attr'];changed=model(**tensors)
        self.assertTrue(torch.equal(original,changed))
        permutation=np.array([0,3,1,2]);inverse=np.argsort(permutation);permuted=deepcopy(g)
        permuted['node_raw']=np.asarray(g['node_raw'])[permutation].tolist()
        permuted['edge_index']=inverse[np.asarray(g['edge_index'])].tolist()
        for kind in ('gnn','mlp'):
            network=build_model(kind,bounds,plan['config']);a,b=predict(network,[g,permuted],plan['normalization'])
            np.testing.assert_allclose(b,np.r_[a[0],a[permutation[1:]],a[-1]],atol=2e-7,rtol=0)

    def test_weighted_loss_ignores_padded_users_in_value_and_gradient(self):
        mask=torch.tensor([[True,True,True,False],[True,True,True,True]])
        prediction=torch.tensor([[.1,.2,.3,99.,.4],[.1,.2,.3,.4,.5]],requires_grad=True)
        target=torch.zeros_like(prediction)
        loss=balanced_allocation_loss(prediction,target,mask,32.,2.,1.)
        expected=((32*.1**2+2*(.2**2+.3**2)/2+.4**2)+(32*.1**2+2*(.2**2+.3**2+.4**2)/3+.5**2))/2/35
        self.assertAlmostEqual(float(loss.detach()),expected,places=7);loss.backward()
        self.assertEqual(float(prediction.grad[0,3]),0.)

    def test_optimizer_resume_matches_uninterrupted_weights_and_history(self):
        plan,rows=small_plan();bounds=fixture()[0]['bounds']
        with tempfile.TemporaryDirectory() as full,tempfile.TemporaryDirectory() as resumed:
            fit(full,'gnn',7,plan,bounds,rows,rows)
            with self.assertRaises(InterruptedError):fit(resumed,'gnn',7,plan,bounds,rows,rows,stop_after_epoch=2)
            self.assertEqual(read_json(Path(resumed)/'fits/gnn_s7/resume.json')['epoch'],2)
            fit(resumed,'gnn',7,plan,bounds,rows,rows)
            a,sa=load_model(Path(full)/'fits/gnn_s7/best_weighted.pt');b,sb=load_model(Path(resumed)/'fits/gnn_s7/best_weighted.pt')
            self.assertEqual(sa['epoch'],sb['epoch']);self.assertEqual(sa['validation_errors'],sb['validation_errors'])
            self.assertTrue(all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict()))
            self.assertEqual((Path(full)/'fits/gnn_s7/history.csv').read_bytes(),(Path(resumed)/'fits/gnn_s7/history.csv').read_bytes())


class PhysicsSelectionTests(unittest.TestCase):
    def test_physical_selection_uses_validation_passes_before_loss(self):
        refs={};rows=[]
        for kind in ('gnn','mlp'):
            for name,loss,passes in (('low_loss',.001,1),('feasible',.01,2)):
                key=kind+'_'+name;refs[key]={'kind':kind,'seed':7,'epoch':20,'validation_errors':{'weighted_loss':loss}}
                for i in range(2):rows.append({'candidate_id':key,'split':'validation','role':'validation','regime':'seen_k',
                    'passed':i<passes,'returned':True,'bob_rate':4.,'gated_bob_rate':4. if i<passes else 0.})
        plan={'config':{'seeds':[7]},'planned_counts':{'validation':2},'physical_selection':'unit'}
        selected=select_checkpoints(refs,rows,plan)
        self.assertEqual(selected['views']['gnn_s7_physical']['candidate_id'],'gnn_feasible')
        self.assertEqual(selected['views']['gnn_s7_weighted_loss']['candidate_id'],'gnn_low_loss')
        bad=deepcopy(rows);bad[0]['split']='test'
        with self.assertRaises(ValueError):select_checkpoints(refs,bad,plan)

    def test_family_counts_include_all_cases_and_views_not_only_successes(self):
        plan={'max_validation_candidates':36,'max_test_views':13,'config':{'alpha':.05},
              'validation_jobs':[{}]*8,'test_jobs':[{}]*10}
        self.assertEqual(confidence(plan,'validation',{'cases':[{}]*8}),(288,.05/16))
        self.assertEqual(confidence(plan,'test',{'cases':[{}]*8}),(104,.05/20))

    def test_no_fading_generation_before_global_fit_prediction_and_selection_barriers(self):
        with tempfile.TemporaryDirectory() as folder,patch('ris_noma.continuous_learning_physics.context') as bank:
            root=Path(folder)
            with self.assertRaisesRegex(ValueError,'Freeze every fit'):audit_task((folder,'validation',{'split':'validation','id':'x'}))
            (root/'fits_frozen.json').write_text('{}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Freeze every prediction'):audit_task((folder,'validation',{'split':'validation','id':'x'}))
            with self.assertRaisesRegex(ValueError,'Freeze validation-only'):audit_task((folder,'test',{'split':'test','id':'x'}))
            bank.assert_not_called()

    def test_fresh_banks_keep_geometric_los_and_use_new_separate_streams(self):
        _,_,_,_,source=fixture();array={'carrier_hz':3.5e9,'spacing_wavelengths':.5,'array_axis':[0.,1.]}
        source_plan={'config':{'profile':'geometry','power_dbm_range':[30.,40.]},
            'deployment':{'id':'reference','alice':[0.,0.],'ris':[95.,5.]},'array_settings':array}
        plan={'config':{'validation_physics_samples':1000,'test_physics_samples':1000},'streams':STREAMS}
        job={'layout_id':2008,'n_public':2}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);write_json(root/'source_7e_plan.json',source_plan)
            write_json(root/'source_physics_plan.json',source);write_json(root/'source_geometry_plan.json',{'config':{'power_dbm':[35.]}})
            _,_,first,a=context(plan,root,job,'validation');_,_,second,b=context(plan,root,job,'test')
        self.assertEqual(a['profile'],'geometry');self.assertEqual(a['receiver_xy'],b['receiver_xy'])
        self.assertEqual(a['los_sha256'],b['los_sha256']);self.assertNotEqual(a['bank_entropy'],b['bank_entropy'])
        self.assertFalse(np.array_equal(first.h_ar,second.h_ar))
        self.assertFalse(set(STREAMS.values()) & set(DATA_STREAMS.values()))

    def test_missing_teacher_case_stays_in_physical_denominator_but_not_mse(self):
        cases=rows_for(2,2)
        for r in cases:r.update(split='test',role='test')
        cases[1].update(has_label=False,allocation=None)
        rows=[];view='gnn_s7_physical'
        for kind,vid,rule in [('gnn',view,'physical'),('pso_label_reference','pso_label_reference','frozen_7e_teacher_label')]:
            for i,r in enumerate(cases):
                rows.append({k:r[k] for k in ('case_id','layout_id','n_public','regime','power_dbm','xi')}|{
                    'view_id':vid,'kind':kind,'seed':7 if kind=='gnn' else None,'selection_rule':rule,
                    'allocation':[.2,.4,.4,.5] if kind=='gnn' else r['allocation'],
                    'returned':kind=='gnn' or r['has_label'],'passed':i==0,'bob_rate':4. if i==0 else 0.,'gated_bob_rate':4. if i==0 else 0.})
        plan={'config':{'loss_weights':{'mb':32.,'public_total':2.,'s':1.}},'power_dbm_range':[30.,40.],'dep_range':[.75,.95]}
        metrics,errors=calculate_metrics(plan,{'views':{view:{}}},cases,rows)
        actual=metrics['by_view'][0]
        self.assertEqual(actual['cases'],2);self.assertEqual(actual['labeled_error_cases'],1)
        self.assertEqual(actual['pass_fraction'],.5);self.assertEqual(actual['gated_rate'],2.)
        self.assertEqual(len(errors),1)

    def test_nonempty_output_and_changed_resume_are_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);write_json(root/'plan.json',{'workers':2})
            with patch('ris_noma.continuous_learning.inspect',return_value=({'workers':4},{},{},[],[])), \
                 patch('ris_noma.continuous_learning.fit') as training:
                with self.assertRaisesRegex(ValueError,'nonempty'):run('source',folder,{},4)
                with self.assertRaisesRegex(ValueError,'Changed'):run('source',folder,{},4,resume=True)
                training.assert_not_called()


if __name__=='__main__':unittest.main()
