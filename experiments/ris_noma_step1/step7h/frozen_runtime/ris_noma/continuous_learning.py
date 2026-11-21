"""7f supervised training on continuous 7e cases, validation-only selection and fresh tests."""
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
from importlib.metadata import version
import csv
import json
import multiprocessing as mp
import platform
from filelock import FileLock
import numpy as np
import scipy
import torch

from .continuous_learning_data import (FORMAT,KINDS,RULES,ROLES,COPIES,validate_config,dataset_info,
    load_cases,labeled,validate_rows,train_normalizer,source_hashes,group_path)
from .continuous_learning_fit import fit,read_fit
from .continuous_learning_models import build_model,load_model
from .continuous_learning_physics import STREAMS,audit_task,read_record,commit,confidence
from .continuous_dataset import check as check_dataset,read_lines
from .expanded_data import write_table
from .learning_data import read_json,sha256
from .learning_train import configure_runtime
from .multik_benchmark import verify_manifest
from .multik_core import physical_allocation
from .multik_data_core import at_target
from .multik_dataset import immutable,immutable_bytes,write_bytes,write_json
from .multik_graph import fit_normalizer
from .multik_models import predict
from .validation import family_constants

ROOT=Path(__file__).resolve().parents[1]
NUMERIC_FILES=('run_step7f.py','ris_noma/continuous_learning_data.py','ris_noma/continuous_learning_models.py',
    'ris_noma/continuous_learning_fit.py','ris_noma/continuous_learning_physics.py','ris_noma/continuous_learning.py')


def inspect(dataset,settings,workers=2):
    if type(workers) is not int or not 1<=workers<=4:raise ValueError('Use 1..4 physics workers; 2 recommended')
    dataset=Path(dataset).resolve();status,prior,source=dataset_info(dataset);c=validate_config(settings)
    train=load_cases(dataset,'train',prior);validation=load_cases(dataset,'validation',prior)
    for name,rows in (('train',train),('validation',validation)):
        validate_rows(rows,name,prior,source['base_config']['bounds'])
        if any(not any(r['n_public']==k for r in labeled(rows)) for k in prior['config']['training_public_counts']):
            raise ValueError('Need at least one labeled case of every trained K in train and validation')
    normalization=train_normalizer(train,read_json(dataset/'graphs/normalization.json'))
    configure_runtime(0,1,'cpu')
    parameters={kind:sum(p.numel() for p in build_model(kind,source['base_config']['bounds'],c).parameters()) for kind in KINDS}
    max_candidates=2*len(c['seeds'])*(len(c['snapshot_epochs'])+1);views=2*len(c['seeds'])*len(RULES)+1
    planned={name:prior['planned_counts'][name] for name in ROLES}
    files=tuple(dict.fromkeys((*prior['code_sha256'],*NUMERIC_FILES)))
    plan={'step':'7f','format':FORMAT,'config':c,'workers':workers,'device':'cpu','threads_per_process':1,
        'dataset_sha256':source_hashes(dataset),'code_sha256':{name:sha256(ROOT/name) for name in files},
        'versions':{'system':platform.system(),'python':platform.python_version(),'numpy':np.__version__,
                    'scipy':scipy.__version__,'torch':str(torch.__version__),'filelock':version('filelock')},
        'source_versions':prior['versions'],'source_verified_files':status['verified_files'],
        'normalization':normalization,'labeled_counts':{name:status['exported_counts'][name] for name in ROLES},
        'planned_counts':planned,'training_public_counts':prior['config']['training_public_counts'],
        'unseen_public_counts':prior['config']['unseen_public_counts'],
        'power_dbm_range':prior['config']['power_dbm_range'],'dep_range':prior['config']['dep_range'],
        'profile':prior['config']['profile'],'geometry_scope':prior['deployment'],
        'validation_jobs':[j for j in prior['jobs'] if j['split']=='validation'],
        'test_jobs':[j for j in prior['jobs'] if j['split']=='test'],
        'streams':STREAMS,'model_parameters':parameters,'max_validation_candidates':max_candidates,
        'max_test_views':views,'planned_validation_rows':max_candidates*planned['validation'],
        'planned_test_rows':views*(planned['test']+planned['test_unseen_k']),
        'normalization_fitted_on':'exported labeled train graphs only; shared across K',
        'loss':'(32-style mb weight * squared error + public_total * mean squared public error + s weight * squared error) / sum_weights; padding ignored',
        'architecture_policy':'unchanged 7c variable-K GNN and parameter-matched shared pointwise pooled MLP; no edge messages in MLP',
        'checkpoint_pool':'fixed snapshots plus best weighted validation loss per fit, including epoch zero if best',
        'physical_selection':'per kind/seed: most validation passes, highest gated Bob rate, lowest weighted validation loss, earlier epoch, ID',
        'weighted_selection':'per kind/seed: minimum weighted validation loss, earlier epoch, ID',
        'deployment_selection':'one physical-selected seed per kind, using validation only',
        'physical_policy':'unchanged 7e geometric LoS, scalar RIS noise, full SIC/QoS/DEP/outage evaluator; fresh bank streams 511/512',
        'confidence_policy':'alpha/(2*phase_group_count); family covers all planned checkpoint/views times cases per group',
        'test_policy':'all fits and validation choices freeze before test predictions/bank; no test checkpoint or seed tuning',
        'missing_label_policy':'no audit filtering; all planned validation/test cases physically evaluated; MSE uses available labels only',
        'physical_gradient':False,'test_used_for_training_or_selection':False,'final_unseen_test':False}
    return plan,source,prior,train,validation


def write_lines(path,rows):
    write_bytes(path,(''.join(json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n' for v in rows)).encode('utf-8'))


def freeze(output,name,paths):
    immutable(Path(output)/(name+'.json'),{'phase':name,'plan_sha256':sha256(Path(output)/'plan.json'),
        'sha256':{p.relative_to(output).as_posix():sha256(p) for p in paths}})


def candidates(output,plan):
    refs={};manifests={}
    for kind in KINDS:
        for seed in plan['config']['seeds']:
            record=read_fit(output,kind,seed,plan);refs.update(record['candidates'])
            path=Path(output)/'fits'/f'{kind}_s{seed}'/'fit_manifest.json'
            manifests[path.relative_to(output).as_posix()]=sha256(path)
    if len(refs)!=plan['max_validation_candidates']:raise ValueError('Wrong declared checkpoint pool size')
    immutable(Path(output)/'fits_frozen.json',{'candidates':refs,'manifests':manifests,'plan_sha256':plan['plan_sha256']})
    return refs


def commit_predictions(output,phase,jobs,rows,views,normalizer,anchor):
    output=Path(output)
    if phase=='test' and not (output/'selection_frozen.json').is_file():raise ValueError('Freeze checkpoint choices before test prediction')
    if not (output/'fits_frozen.json').is_file():raise ValueError('Freeze all fits before prediction')
    models={}
    for view in views.values():
        if view['candidate_id'] not in models:models[view['candidate_id']]=load_model(output/view['path'])[0]
    paths=[]
    for job in jobs:
        path=group_path(output,phase,'predictions',job);cached=read_record(output,phase,'predictions',job)
        if cached is None:
            subset=[r for r in rows if (r['layout_id'],r['n_public'])==(job['layout_id'],job['n_public'])]
            if len(subset)!=len(job['cases']):raise ValueError('Missing planned graph inputs')
            proposals=[];cache={}
            keys=('case_id','layout_id','n_public','split','regime','role','power_dbm','xi','order')
            for view_id,view in views.items():
                key=view['candidate_id']
                if key not in cache:cache[key]=predict(models[key],[r['graph'] for r in subset],normalizer)
                for row,y in zip(subset,cache[key]):
                    proposals.append({name:row[name] for name in keys}|{'view_id':view_id,'candidate_id':key,
                        'kind':view['kind'],'seed':view['seed'],'selection_rule':view.get('selection_rule','candidate_pool'),
                        'allocation':y.tolist(),'returned':True})
            if phase=='test':
                for row in subset:
                    proposals.append({name:row[name] for name in keys}|{'view_id':'pso_label_reference',
                        'candidate_id':'pso_label_reference','kind':'pso_label_reference','seed':None,
                        'selection_rule':'frozen_7e_teacher_label','allocation':row['allocation'],'returned':row['has_label']})
            commit(path,{'job':job,'phase':phase,'rows':proposals,'anchor_sha256':anchor,
                         'plan_sha256':sha256(output/'plan.json')})
        elif cached['anchor_sha256']!=anchor:raise ValueError('Predictions belong to another frozen choice')
        paths.append(path)
    freeze(output,phase+'_predictions_frozen',paths)


def audit_phase(output,phase,jobs,workers):
    pending=[job for job in jobs if read_record(output,phase,'audit',job) is None]
    done=len(jobs)-len(pending);print(f'7f {phase} physics: {done}/{len(jobs)} groups committed',flush=True)
    if workers==1:
        for job in pending:
            name=audit_task((str(output),phase,job));done+=1;print(f'7f {phase} physics {done}/{len(jobs)}: {name}',flush=True)
    elif pending:
        executor=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'));active={};queue=iter(pending)
        def submit():
            job=next(queue,None)
            if job is not None:active[executor.submit(audit_task,(str(output),phase,job))]=job
        try:
            for _ in range(workers):submit()
            while active:
                finished,_=wait(active,return_when=FIRST_COMPLETED)
                for future in finished:
                    name=future.result();del active[future];done+=1
                    print(f'7f {phase} physics {done}/{len(jobs)}: {name}',flush=True);submit()
        finally:
            for future in active:future.cancel()
            executor.shutdown(wait=True,cancel_futures=True)
    paths=[group_path(output,phase,'audit',job) for job in jobs]
    if any(read_record(output,phase,'audit',job) is None for job in jobs):raise ValueError('Incomplete physics phase')
    freeze(output,phase+'_audit_frozen',paths)


def phase_rows(output,phase,jobs):return [r for j in jobs for r in read_record(output,phase,'audit',j)['rows']]


def aggregate(rows):
    if not rows:raise ValueError('Cannot aggregate empty physical cases')
    return {'cases':len(rows),'returned':sum(r['returned'] for r in rows),'passed':sum(r['passed'] for r in rows),
        'pass_fraction':float(np.mean([r['passed'] for r in rows])),
        'bob_rate':float(np.mean([r['bob_rate'] for r in rows])),
        'gated_rate':float(np.mean([r['gated_bob_rate'] for r in rows]))}


def select_checkpoints(refs,rows,plan):
    if not rows or any(r['split']!='validation' or r['role']!='validation' or r['regime']!='seen_k' for r in rows):
        raise ValueError('Only seen-K validation may select checkpoints')
    scores={}
    for key in refs:
        part=[r for r in rows if r['candidate_id']==key]
        if len(part)!=plan['planned_counts']['validation']:raise ValueError('Incomplete physical validation candidate')
        scores[key]=aggregate(part)
    def physical_key(key):
        return (-scores[key]['passed'],-scores[key]['gated_rate'],refs[key]['validation_errors']['weighted_loss'],refs[key]['epoch'],key)
    views={}
    for kind in KINDS:
        for seed in plan['config']['seeds']:
            keys=[key for key,ref in refs.items() if ref['kind']==kind and ref['seed']==seed]
            if not keys:raise ValueError('Missing kind/seed in checkpoint pool')
            chosen={'weighted_loss':min(keys,key=lambda key:(refs[key]['validation_errors']['weighted_loss'],refs[key]['epoch'],key)),
                    'physical':min(keys,key=physical_key)}
            for rule,key in chosen.items():
                views[f'{kind}_s{seed}_{rule}']={**refs[key],'candidate_id':key,'selection_rule':rule,'validation_physics':scores[key]}
    deployment={}
    for kind in KINDS:
        keys=[key for key,v in views.items() if v['kind']==kind and v['selection_rule']=='physical']
        chosen=min(keys,key=lambda key:(*physical_key(views[key]['candidate_id'])[:3],views[key]['seed']))
        deployment[kind]={'view_id':chosen,**views[chosen]}
    return {'views':views,'deployment':deployment,'candidate_validation_scores':scores,
        'test_used_for_selection':False,'selection_policy':plan['physical_selection']}


def allocation_errors(plan,cases,rows):
    truth={r['case_id']:r['allocation'] for r in cases if r['has_label']};w=plan['config']['loss_weights'];out=[]
    for row in rows:
        if row['kind']=='pso_label_reference' or row['case_id'] not in truth:continue
        delta=np.asarray(row['allocation'])-np.asarray(truth[row['case_id']])
        out.append({key:row[key] for key in ('view_id','kind','seed','selection_rule','case_id','layout_id','n_public','regime','power_dbm','xi')}|{
            'case_mse':float(np.mean(delta**2)),
            'weighted_loss':float((w['mb']*delta[0]**2+w['public_total']*np.mean(delta[1:-1]**2)+w['s']*delta[-1]**2)/sum(w.values())),
            'mb_mae':float(abs(delta[0])),'public_mae':float(np.mean(abs(delta[1:-1]))),'s_mae':float(abs(delta[-1]))})
    return out


def calculate_metrics(plan,selection,cases,rows):
    errors=allocation_errors(plan,cases,rows)
    def measures(part,err):
        fields=('case_mse','weighted_loss','mb_mae','public_mae','s_mae')
        return aggregate(part)|{'labeled_error_cases':len(err)}|{k:float(np.mean([r[k] for r in err])) if err else None for k in fields}
    by_view=[];by_regime=[];by_k=[];by_dep=[];by_power=[]
    for view_id in (*selection['views'],'pso_label_reference'):
        part=[r for r in rows if r['view_id']==view_id];err=[r for r in errors if r['view_id']==view_id]
        if not part:raise ValueError('Missing test view')
        meta={key:part[0][key] for key in ('view_id','kind','seed','selection_rule')}
        by_view.append(meta|measures(part,err))
        for regime in ('seen_k','unseen_k'):
            p=[r for r in part if r['regime']==regime];e=[r for r in err if r['regime']==regime]
            if p:by_regime.append(meta|{'regime':regime}|measures(p,e))
        for k in sorted({r['n_public'] for r in part}):
            p=[r for r in part if r['n_public']==k];e=[r for r in err if r['n_public']==k]
            by_k.append(meta|{'n_public':k,'regime':p[0]['regime']}|measures(p,e))
        for field,domain,target in (('xi','dep_range',by_dep),('power_dbm','power_dbm_range',by_power)):
            edges=np.linspace(*plan[domain],5)
            for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
                p=[r for r in part if lo<=r[field] and (r[field]<=hi if i==3 else r[field]<hi)]
                if p:target.append(meta|{'bin_low':float(lo),'bin_high':float(hi),'last_bin_closed':i==3}|aggregate(p))
    pooled=[]
    for kind in KINDS:
        for rule in RULES:
            for regime,source in (('all',by_view),('seen_k',by_regime),('unseen_k',by_regime)):
                group=[r for r in source if r['kind']==kind and r['selection_rule']==rule and (regime=='all' or r['regime']==regime)]
                if not group:continue
                value={'kind':kind,'selection_rule':rule,'regime':regime,'training_seeds':len(group),'cases_per_seed':group[0]['cases']}
                for field in ('pass_fraction','gated_rate','case_mse'):
                    data=[r[field] for r in group if r[field] is not None]
                    value[field+'_mean']=float(np.mean(data)) if data else None
                    value[field+'_std']=float(np.std(data,ddof=1)) if len(data)>1 else 0. if data else None
                pooled.append(value)
    return {'by_view':by_view,'by_regime':by_regime,'by_k':by_k,'by_dep_bin':by_dep,'by_power_bin':by_power,'across_seeds':pooled},errors


def export_metrics(output,metrics,errors,rows):
    output=Path(output)
    if errors:write_table(output/'test_allocation_errors.csv',errors)
    else:
        with (output/'test_allocation_errors.csv').open('w',encoding='utf-8-sig',newline='') as handle:
            csv.writer(handle).writerow(['view_id','kind','seed','selection_rule','case_id','layout_id','n_public','regime','power_dbm','xi','case_mse','weighted_loss','mb_mae','public_mae','s_mae'])
    for name,key in (('test_metrics.csv','by_view'),('test_by_regime.csv','by_regime'),('test_by_k.csv','by_k'),
                     ('test_by_dep_bin.csv','by_dep_bin'),('test_by_power_bin.csv','by_power_bin'),('test_seed_summary.csv','across_seeds')):
        write_table(output/name,metrics[key])
    physical=[{key:r[key] for key in ('view_id','kind','seed','selection_rule','case_id','layout_id','n_public','regime','power_dbm','xi','returned','passed','bob_rate','gated_bob_rate','failed_checks')}
        |{'dep_lower':r.get('dep_lower'),'qos_lower_margin':r.get('qos_lower_margin'),'bob_outage_upper':r.get('bob_outage_upper')} for r in rows]
    write_table(output/'test_physics_rows.csv',physical)


def run(dataset,output,settings,workers=2,resume=False):
    dataset,output=Path(dataset).resolve(),Path(output).resolve()
    if dataset==output or dataset in output.parents or output in dataset.parents:raise ValueError('Use a separate non-nested 7f output')
    output.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(output)+'.lock',timeout=0):
        plan,source,prior,train,validation=inspect(dataset,settings,workers)
        if output.exists() and any(output.iterdir()):
            if not resume:raise ValueError('Output is nonempty; use --resume with the same plan or a new output')
            if read_json(output/'plan.json')!=plan:raise ValueError('Changed config, code, dataset, workers or runtime; use a new output')
            if (output/'completion.json').exists():return check(output,dataset)
        output.mkdir(exist_ok=True);immutable(output/'plan.json',plan);immutable(output/'normalization.json',plan['normalization'])
        for original,copy in COPIES.items():immutable_bytes(output/copy,(dataset/original).read_bytes())
        for name in plan['code_sha256']:immutable_bytes(output/'source_snapshot'/name,(ROOT/name).read_bytes())
        for name,rows in (('train',train),('validation',validation)):immutable_bytes(output/'inputs'/(name+'.jsonl'),(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows)).encode('utf-8'))
        active=dict(plan,plan_sha256=sha256(output/'plan.json'))
        for kind in KINDS:
            for seed in plan['config']['seeds']:fit(output,kind,seed,active,source['base_config']['bounds'],labeled(train),labeled(validation))
        verify_manifest(ROOT,plan['code_sha256']);refs=candidates(output,active)
        views={key:dict(ref,candidate_id=key) for key,ref in refs.items()}
        commit_predictions(output,'validation',plan['validation_jobs'],validation,views,plan['normalization'],sha256(output/'fits_frozen.json'))
        audit_phase(output,'validation',plan['validation_jobs'],workers)
        val_rows=phase_rows(output,'validation',plan['validation_jobs']);selection=select_checkpoints(refs,val_rows,plan)
        immutable(output/'selection_frozen.json',selection|{'plan_sha256':active['plan_sha256'],
            'fits_frozen_sha256':sha256(output/'fits_frozen.json'),
            'validation_audit_frozen_sha256':sha256(output/'validation_audit_frozen.json')})
        # Test targets/graphs are loaded for model evaluation only after choices freeze.
        tests=[]
        for name in ('test','test_unseen_k'):
            rows=load_cases(dataset,name,prior);validate_rows(rows,name,prior,source['base_config']['bounds']);tests+=rows
            immutable_bytes(output/'inputs'/(name+'.jsonl'),(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows)).encode('utf-8'))
        commit_predictions(output,'test',plan['test_jobs'],tests,selection['views'],plan['normalization'],sha256(output/'selection_frozen.json'))
        audit_phase(output,'test',plan['test_jobs'],workers);test_rows=phase_rows(output,'test',plan['test_jobs'])
        metrics,errors=calculate_metrics(plan,selection,tests,test_rows);export_metrics(output,metrics,errors,test_rows)
        check_dataset(dataset)
        if source_hashes(dataset)!=plan['dataset_sha256']:raise ValueError('Source 7e changed during training/evaluation')
        verify_manifest(ROOT,plan['code_sha256'])
        result={'step':'7f','complete':True,'scope':plan['config']['scope'],'format':FORMAT,
            'labeled_counts':plan['labeled_counts'],'planned_counts':plan['planned_counts'],
            'fits':2*len(plan['config']['seeds']),'candidate_checkpoints':len(refs),'selected_views':len(selection['views']),
            'model_parameters':plan['model_parameters'],'validation_rows':len(val_rows),'test_rows':len(test_rows),
            'metrics':metrics,'deployment':selection['deployment'],'test_used_for_training_or_selection':False,
            'physical_gradient':False,'source_7e_unchanged':True,'final_unseen_test':False,
            'interpretation':'Development pilot: direct GNN/pooled MLP and frozen PSO teacher labels; no short-PSO hybrid or latency benchmark yet.'}
        write_json(output/'summary.json',result)
        files=[p for p in sorted(output.rglob('*')) if p.is_file() and p.name!='completion.json' and not p.name.endswith('.tmp') and 'figures' not in p.relative_to(output).parts]
        immutable(output/'completion.json',{'sha256':{p.relative_to(output).as_posix():sha256(p) for p in files}})
        return check(output,dataset)


def check(output,dataset=None):
    output=Path(output).resolve();manifest=read_json(output/'completion.json')['sha256'];verify_manifest(output,manifest)
    plan=read_json(output/'plan.json');validate_config(plan['config'])
    if plan['format']!=FORMAT or plan['step']!='7f' or plan['streams']!=STREAMS:raise ValueError('Wrong 7f format/policy')
    verify_manifest(ROOT,plan['code_sha256']);verify_manifest(output/'source_snapshot',plan['code_sha256'])
    active=dict(plan,plan_sha256=sha256(output/'plan.json'));prior=read_json(output/'source_7e_plan.json')
    physics=read_json(output/'source_physics_plan.json');bounds=physics['base_config']['bounds']
    for name,copy in COPIES.items():
        if sha256(output/copy)!=plan['dataset_sha256'][name]:raise ValueError('Copied 7e provenance differs')
    if any(plan['code_sha256'].get(k)!=v for k,v in prior['code_sha256'].items()):raise ValueError('Inherited 7e numerical code changed')
    inputs={name:read_lines(output/'inputs'/(name+'.jsonl')) for name in ROLES}
    for name,rows in inputs.items():
        validate_rows(rows,name,prior,bounds)
        if len(rows)!=plan['planned_counts'][name] or len(labeled(rows))!=plan['labeled_counts'][name]:raise ValueError('Input sample denominator changed')
    norm=train_normalizer(inputs['train'],read_json(output/'source_normalization.json'))
    if norm!=plan['normalization'] or norm!=read_json(output/'normalization.json'):raise ValueError('Normalizer is not train-only')
    refs={};fit_hashes={}
    for kind in KINDS:
        for seed in plan['config']['seeds']:
            record=read_fit(output,kind,seed,active);refs.update(record['candidates'])
            fit_name=f'fits/{kind}_s{seed}/fit_manifest.json';fit_hashes[fit_name]=sha256(output/fit_name)
            if record['test_used_for_training_or_selection'] or record['physical_gradient']:raise ValueError('Unexpected fit policy')
            history=list(csv.DictReader((output/'fits'/f'{kind}_s{seed}'/'history.csv').open(encoding='utf-8-sig',newline='')))
            if [int(r['epoch']) for r in history]!=list(range(plan['config']['epochs']+1)):raise ValueError('Incomplete training history')
            best=min(history,key=lambda r:(float(r['validation_weighted_loss']),int(r['epoch'])))
            if record['best_weighted_epoch']!=int(best['epoch']):raise ValueError('Wrong best validation-loss epoch')
            for ref in record['candidates'].values():
                item=history[ref['epoch']]
                if any(ref['validation_errors'][k]!=float(item['validation_'+k]) for k in ref['validation_errors']):
                    raise ValueError('Checkpoint loss differs from its recorded validation epoch')
    frozen=read_json(output/'fits_frozen.json')
    if frozen['candidates']!=refs or frozen['manifests']!=fit_hashes or frozen['plan_sha256']!=active['plan_sha256'] or len(refs)!=plan['max_validation_candidates']:
        raise ValueError('Frozen candidate pool differs')
    selected=read_json(output/'selection_frozen.json')
    for phase in ('validation','test'):
        jobs=plan[phase+'_jobs'];phase_inputs=inputs['validation'] if phase=='validation' else inputs['test']+inputs['test_unseen_k']
        lookup={r['case_id']:r for r in phase_inputs}
        views={key:dict(ref,candidate_id=key) for key,ref in refs.items()} if phase=='validation' else selected['views']|{'pso_label_reference':{'candidate_id':'pso_label_reference'}}
        for stage in ('predictions','audit'):
            f=read_json(output/(phase+'_'+stage+'_frozen.json'));expected={group_path(output,phase,stage,j).relative_to(output).as_posix() for j in jobs}
            if f['plan_sha256']!=active['plan_sha256'] or set(f['sha256'])!=expected or any(manifest.get(k)!=v for k,v in f['sha256'].items()):
                raise ValueError('Frozen physical group set differs')
        for job in jobs:
            predictions=read_record(output,phase,'predictions',job);audit=read_record(output,phase,'audit',job)
            if predictions['job']!=job or audit['job']!=job:raise ValueError('Wrong physical job')
            if predictions['anchor_sha256']!=sha256(output/('fits_frozen.json' if phase=='validation' else 'selection_frozen.json')):
                raise ValueError('Prediction anchor differs')
            if audit['predictions_frozen_sha256']!=sha256(output/(phase+'_predictions_frozen.json')):raise ValueError('Audit prediction barrier differs')
            family,alpha=confidence(plan,phase,job);samples=plan['config'][phase+'_physics_samples'];meta=audit['metadata']
            if len(predictions['rows'])!=family or len(audit['rows'])!=family or audit['family_size']!=family or audit['alpha_per_group']!=alpha:
                raise ValueError('Physical family/denominator changed')
            if meta['bank_samples']!=samples or meta['bank_entropy']!=[prior['root_seed'],job['layout_id'],job['n_public'],STREAMS[phase]] or meta['profile']!=prior['config']['profile']:
                raise ValueError('Wrong physical bank/LoS profile')
            seen=set()
            for before,after in zip(predictions['rows'],audit['rows']):
                if any(after[k]!=v for k,v in before.items()):raise ValueError('Physical audit changed a frozen prediction')
                key=(before['view_id'],before['case_id'])
                if key in seen or before['view_id'] not in views:raise ValueError('Duplicate or undeclared prediction')
                seen.add(key);case=lookup.get(before['case_id'])
                if case is None or any(before[k]!=case[k] for k in ('layout_id','n_public','split','regime','role','power_dbm','xi','order')):
                    raise ValueError('Prediction input/continuous parameters changed')
                if before['candidate_id']!=views[before['view_id']]['candidate_id']:raise ValueError('View uses another checkpoint')
                if before['kind']!='pso_label_reference':
                    ref=views[before['view_id']]
                    if before['kind']!=ref['kind'] or before['seed']!=ref['seed'] or before['selection_rule']!=ref.get('selection_rule','candidate_pool'):
                        raise ValueError('View kind/seed/selection rule differs')
                if before['kind']=='pso_label_reference':
                    if before['allocation']!=case['allocation'] or before['returned']!=case['has_label']:raise ValueError('Teacher reference changed')
                elif not before['returned']:raise ValueError('Model prediction unexpectedly missing')
                if not before['returned']:
                    if after['passed'] or after['gated_bob_rate']!=0. or after['failed_checks']!=['teacher_label_missing']:raise ValueError('Missing reference not counted as failure')
                else:
                    physical_allocation(before['allocation'],job['n_public'],bounds)
                    decision=at_target(after,after['xi'])
                    if any(after[k]!=v for k,v in decision.items()) or after['constants']!=family_constants(samples,family,job['n_public'],alpha):
                        raise ValueError('Physical decision/confidence constants changed')
            expected={(view_id,r['case_id']) for view_id in views for r in phase_inputs if (r['layout_id'],r['n_public'])==(job['layout_id'],job['n_public'])}
            if seen!=expected:raise ValueError('Incomplete checkpoint/case Cartesian product')
    val_rows=phase_rows(output,'validation',plan['validation_jobs']);expected_selection=select_checkpoints(refs,val_rows,plan)
    if (any(selected[k]!=v for k,v in expected_selection.items()) or selected['plan_sha256']!=active['plan_sha256']
            or selected['fits_frozen_sha256']!=sha256(output/'fits_frozen.json')
            or selected['validation_audit_frozen_sha256']!=sha256(output/'validation_audit_frozen.json')):
        raise ValueError('Checkpoint choice differs from validation-only rule')
    test_rows=phase_rows(output,'test',plan['test_jobs']);metrics,_=calculate_metrics(plan,selected,inputs['test']+inputs['test_unseen_k'],test_rows)
    result=read_json(output/'summary.json')
    scalar={'step':'7f','complete':True,'scope':plan['config']['scope'],'format':FORMAT,
        'labeled_counts':plan['labeled_counts'],'planned_counts':plan['planned_counts'],
        'fits':2*len(plan['config']['seeds']),'candidate_checkpoints':len(refs),'selected_views':len(selected['views']),
        'model_parameters':plan['model_parameters'],'validation_rows':len(val_rows),'test_rows':len(test_rows),
        'metrics':metrics,'deployment':selected['deployment'],'test_used_for_training_or_selection':False,
        'physical_gradient':False,'source_7e_unchanged':True,'final_unseen_test':False}
    if any(result[k]!=v for k,v in scalar.items()) or len(val_rows)!=plan['planned_validation_rows'] or len(test_rows)!=plan['planned_test_rows']:
        raise ValueError('Summary/budget differs from frozen rows')
    if dataset is not None:
        check_dataset(dataset)
        if source_hashes(dataset)!=plan['dataset_sha256']:raise ValueError('Source 7e changed')
        for name in ROLES:
            if load_cases(dataset,name,prior)!=inputs[name]:raise ValueError('Learning inputs differ from original 7e records')
    return result|{'verified_files':len(manifest)}


def status(output):
    output=Path(output);plan=read_json(output/'plan.json')
    return {'step':'7f','scope':plan['config']['scope'],'complete':(output/'completion.json').exists(),
        'completed_fits':sum((output/'fits'/f'{kind}_s{seed}'/'fit_manifest.json').exists() for kind in KINDS for seed in plan['config']['seeds']),
        'resume_epochs':{f'{kind}_s{seed}':read_json(output/'fits'/f'{kind}_s{seed}'/'resume.json')['epoch']
            for kind in KINDS for seed in plan['config']['seeds'] if (output/'fits'/f'{kind}_s{seed}'/'resume.json').exists()},
        'physics_groups':{phase:sum(read_record(output,phase,'audit',j) is not None for j in plan[phase+'_jobs']) for phase in ('validation','test')}}
