"""7f fresh validation/test banks: geometric LoS and unchanged scalar RIS physics."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from .channels import build_bank
from .continuous_data import geometry_from
from .learning_data import read_json,sha256
from .multik_data_core import at_target
from .multik_dataset import read_committed,receipt_path,write_json
from .stable_assessment import raw_assessment

STREAMS={'validation':511,'test':512}


def stage_path(output,phase,stage,job):return Path(output)/f'{phase}_{stage}'/(job['id']+'.json')


def commit(path,value):
    write_json(path,value);write_json(receipt_path(path),{'sha256':sha256(path)})


def read_record(output,phase,stage,job):
    row=read_committed(stage_path(output,phase,stage,job))
    if row is not None and row.get('plan_sha256')!=sha256(Path(output)/'plan.json'):
        raise ValueError('Committed 7f group belongs to another plan')
    return row


def confidence(plan,phase,job):
    pool=plan['max_validation_candidates'] if phase=='validation' else plan['max_test_views']
    return pool*len(job['cases']),plan['config']['alpha']/(2*len(plan[phase+'_jobs']))


def context(plan,output,job,phase):
    if phase not in STREAMS or plan['streams']!=STREAMS:raise ValueError('Unknown 7f bank role')
    output=Path(output);source_plan=read_json(output/'source_7e_plan.json')
    physics=read_json(output/'source_physics_plan.json');geometry=read_json(output/'source_geometry_plan.json')
    samples=plan['config'][phase+'_physics_samples']
    c,layout,meta=geometry_from(source_plan,physics,geometry,job,samples)
    entropy=[c['root_seed'],job['layout_id'],job['n_public'],STREAMS[phase]]
    full=build_bank(c,layout,np.random.default_rng(np.random.SeedSequence(entropy)))
    names=('h_ar','h_tilde','g_scalar_tilde','signal_gain_tilde','noise_gain_tilde','mean_h_ar_power')
    bank=SimpleNamespace(**{name:getattr(full,name) for name in names})
    meta.update(bank_role=phase,bank_entropy=entropy,bank_samples=samples)
    return c,layout,bank,meta


def audit_task(arguments):
    folder,phase,job=arguments;output=Path(folder)
    if phase not in STREAMS or job['split']!=phase:raise ValueError('Wrong physical split')
    # These barriers run before even generating fading samples.
    if not (output/'fits_frozen.json').is_file():raise ValueError('Freeze every fit before physical validation')
    if phase=='test' and not (output/'selection_frozen.json').is_file():
        raise ValueError('Freeze validation-only checkpoint choices before test bank generation')
    if not (output/(phase+'_predictions_frozen.json')).is_file():
        raise ValueError('Freeze every prediction before generating its physical bank')
    cached=read_record(output,phase,'audit',job)
    if cached is not None:return job['id']
    plan=read_json(output/'plan.json');proposals=read_record(output,phase,'predictions',job)
    if proposals is None:raise ValueError('Missing committed model predictions')
    family,alpha=confidence(plan,phase,job)
    if len(proposals['rows'])!=family:raise ValueError('Physical family differs from all planned views/cases')
    c,layout,bank,meta=context(plan,output,job,phase)
    rows=[];cache={}
    for proposal in proposals['rows']:
        if not proposal['returned']:
            if proposal['kind']!='pso_label_reference' or proposal['allocation'] is not None:
                raise ValueError('Only absent teacher labels may be no-return in this step')
            rows.append({**proposal,'passed':False,'bob_rate':0.,'gated_bob_rate':0.,
                         'failed_checks':['teacher_label_missing']})
            continue
        key=(proposal['power_dbm'],tuple(proposal['allocation']))
        if key not in cache:
            cache[key]=raw_assessment(dict(c,total_dbm=proposal['power_dbm']),layout,bank,
                np.asarray(proposal['order'],int),proposal['allocation'],family,alpha)
        raw=cache[key];rows.append({**proposal,**raw,**at_target(raw,proposal['xi'])})
    result={'job':job,'phase':phase,'rows':rows,'unique_physical_evaluations':len(cache),
        'metadata':meta,'family_size':family,'alpha_per_group':alpha,
        'predictions_frozen_sha256':sha256(output/(phase+'_predictions_frozen.json')),
        'plan_sha256':sha256(output/'plan.json'),'test_used_for_selection':False}
    commit(stage_path(output,phase,'audit',job),result)
    return job['id']
