"""7e continuous scenarios; original scalar RIS physics is unchanged."""
from copy import deepcopy
from types import SimpleNamespace
import time
import numpy as np

from .channels import build_bank
from .config import dbm_to_w, validate_config
from .geometry_los import apply_profile
from .multik_core import MultiKObjective
from .multik_data_core import geometry
from .learning_data import read_json

FORMAT = 'continuous_pso_graph_dataset_7e_v1'
STREAMS = {'search':401, 'pso':402, 'selection':403, 'audit':404, 'parameters':405}
ARCHIVE_SIZE = 5
FIELDS = {'scope','training_public_counts','unseen_public_counts','splits','power_dbm_range','dep_range',
    'cases_per_layout','parameter_seed','search_dep_margin','search_qos_margin','search_samples',
    'selection_samples','audit_samples','pso_seeds','pso_evaluations','alpha','profile'}

def validate_settings(settings):
    c=deepcopy(settings)
    if set(c)!=FIELDS or c['scope'] not in ('development_pilot','smoke'):
        raise ValueError('Unexpected 7e config fields/scope')
    for key in ('training_public_counts','unseen_public_counts'):
        v=c[key]
        if not isinstance(v,list) or not v or any(type(x) is not int or x<1 for x in v) or v!=sorted(set(v)):
            raise ValueError('Use sorted distinct positive public counts')
    if set(c['training_public_counts']) & set(c['unseen_public_counts']):
        raise ValueError('Unseen K cannot occur in train/validation')
    if not isinstance(c['splits'],dict) or set(c['splits'])!={'train','validation','test'}:
        raise ValueError('Declare train/validation/test layout groups')
    ids=[]
    for v in c['splits'].values():
        if not isinstance(v,list) or not v or any(type(x) is not int or x<1000 for x in v) or v!=sorted(set(v)):
            raise ValueError('Use nonempty sorted layout IDs >=1000')
        ids+=v
    if len(ids)!=len(set(ids)):raise ValueError('A layout cannot cross splits')
    for key in ('power_dbm_range','dep_range'):
        v=c[key]
        if not isinstance(v,list) or len(v)!=2 or any(type(x) not in (int,float) or not np.isfinite(x) for x in v) or v[0]>=v[1]:
            raise ValueError('Use increasing finite two-element ranges')
    if not 0<c['dep_range'][0]<c['dep_range'][1]<1:raise ValueError('DEP range must be inside (0,1)')
    for key in ('search_dep_margin','search_qos_margin'):
        if type(c[key]) not in (int,float) or not np.isfinite(c[key]) or c[key]<0:raise ValueError('Nonnegative finite margins required')
    if c['dep_range'][1]+c['search_dep_margin']>=1:raise ValueError('DEP plus margin must stay below one')
    for key in ('search_samples','selection_samples','audit_samples'):
        if type(c[key]) is not int or c[key]<1000:raise ValueError('Use at least 1000 samples per bank')
    if type(c['cases_per_layout']) is not int or not 1<=c['cases_per_layout']<=1000:raise ValueError('Invalid cases_per_layout')
    if type(c['parameter_seed']) is not int or c['parameter_seed']<0:raise ValueError('Invalid parameter seed')
    if not isinstance(c['pso_seeds'],list) or not c['pso_seeds'] or any(type(x) is not int or x<0 for x in c['pso_seeds']) or len(set(c['pso_seeds']))!=len(c['pso_seeds']):
        raise ValueError('Use distinct nonnegative PSO seeds')
    if type(c['pso_evaluations']) is not int or c['pso_evaluations']<30 or c['pso_evaluations']%30:
        raise ValueError('PSO budget includes initialization and must be a multiple of 30')
    if type(c['alpha']) not in (int,float) or not np.isfinite(c['alpha']) or not 0<c['alpha']<1:
        raise ValueError('Require finite 0<alpha<1')
    if c['profile'] not in ('geometry','geometry_friis'):raise ValueError('7e uses geometry-dependent LoS')
    return c

def parameter_cases(c,root_seed,layout_id):
    """Seeded randomized Latin hypercube; uniform marginals in dBm and DEP.

    One schedule per layout, shared by K; no rounding or old-grid lookup.
    """
    rng=np.random.default_rng(np.random.SeedSequence([root_seed,layout_id,STREAMS['parameters'],c['parameter_seed']]))
    n=c['cases_per_layout'];unit=np.column_stack([(rng.permutation(n)+rng.random(n))/n for _ in range(2)])
    return [{'case_index':i,'power_dbm':float(c['power_dbm_range'][0]+np.ptp(c['power_dbm_range'])*unit[i,0]),
             'xi':float(c['dep_range'][0]+np.ptp(c['dep_range'])*unit[i,1])} for i in range(n)]

def role(job):
    if job['split']=='query':return 'query'
    return 'test_unseen_k' if job['regime']=='unseen_k' else job['split']

def jobs_for(c,root_seed,query=None):
    if query is not None:
        return [{'id':f"query_l{query['layout_id']:06d}_k{query['n_public']}", 'layout_id':query['layout_id'],
                 'n_public':query['n_public'],'split':'query','regime':'query',
                 'cases':[{'case_index':0,'power_dbm':query['power_dbm'],'xi':query['xi']}]}]
    return [{'id':f'{split}_l{lid:06d}_k{k}','layout_id':lid,'n_public':k,'split':split,
             'regime':'seen_k' if k in c['training_public_counts'] else 'unseen_k',
             'cases':parameter_cases(c,root_seed,lid)}
        for split in ('train','validation','test') for lid in c['splits'][split]
        for k in sorted(c['training_public_counts']+(c['unseen_public_counts'] if split=='test' else []))]

def validate_query(c,query,forbidden_ids):
    q=deepcopy(query)
    if set(q)!={'layout_id','n_public','power_dbm','xi'}:raise ValueError('Unexpected query fields')
    if type(q['layout_id']) is not int or q['layout_id']<1000 or q['layout_id'] in forbidden_ids:
        raise ValueError('Query requires a fresh layout ID outside source and dataset splits')
    if type(q['n_public']) is not int or q['n_public'] not in c['training_public_counts']+c['unseen_public_counts']:
        raise ValueError('Query K must be among declared pilot public counts')
    for key,domain in (('power_dbm','power_dbm_range'),('xi','dep_range')):
        v=q[key]
        if type(v) not in (int,float) or not np.isfinite(v) or not c[domain][0]<=v<=c[domain][1]:
            raise ValueError('Query outside declared pilot domain; no clipping or nearest-grid substitution')
    return q

def geometry_from(plan,source,dataset,job,samples):
    source=deepcopy(source)
    source['base_config']['positions'].update(alice=plan['deployment']['alice'],ris=plan['deployment']['ris'])
    c,layout=geometry(dataset,source,job,samples)
    c,layout,metadata=apply_profile(c,layout,plan['config']['profile'],plan['deployment'],plan['array_settings'])
    for power in plan['config']['power_dbm_range']:validate_config(dict(c,total_dbm=power))
    return c,layout,metadata

def physical_context(plan,output,job,phase,with_bank=True):
    if phase not in ('search','selection','audit'):raise ValueError('Unknown bank role')
    from pathlib import Path
    output=Path(output);source=read_json(output/'source_physics_plan.json')
    dataset=read_json(output/'source_dataset_plan.json');settings=plan['config']
    n=settings[{'search':'search_samples','selection':'selection_samples','audit':'audit_samples'}[phase]]
    c,layout,metadata=geometry_from(plan,source,dataset,job,n)
    if not with_bank:return c,layout,metadata
    entropy=[c['root_seed'],job['layout_id'],job['n_public'],STREAMS[phase]]
    started=time.perf_counter();full=build_bank(c,layout,np.random.default_rng(np.random.SeedSequence(entropy)))
    bank=SimpleNamespace(**{key:getattr(full,key) for key in
        ('h_ar','h_tilde','g_scalar_tilde','signal_gain_tilde','noise_gain_tilde','mean_h_ar_power')})
    metadata.update(bank_role=phase,bank_entropy=entropy,bank_samples=n,bank_generation_seconds=time.perf_counter()-started)
    return c,layout,bank,metadata

def pso_rng(root_seed,job,seed,index):
    return np.random.default_rng(np.random.SeedSequence([root_seed,job['layout_id'],job['n_public'],STREAMS['pso'],seed,index]))

class MarginObjective(MultiKObjective):
    """Only the search QoS target changes; all SINR/rate/DEP arithmetic is reused."""
    def __init__(self,c,layout,bank,order,qos_margin):
        safe=deepcopy(c)
        safe['requirements']['qos_public']=[float(x+qos_margin) for x in c['requirements']['qos_public']]
        super().__init__(safe,layout,bank,order)
