"""7e immutable continuous-parameter dataset, grouped splits and replay checks."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from importlib.metadata import version
import json
import multiprocessing as mp
import platform
import numpy as np
import scipy
from filelock import FileLock

from .continuous_data import (FORMAT,STREAMS,ARCHIVE_SIZE,validate_settings,jobs_for,role,validate_query,geometry_from)
from .continuous_workers import search_job,select_job,audit_job,read_group,choose_label
from .config import dbm_to_w
from .expanded_data import write_table
from .graph_data import NODE_FEATURES,GLOBAL_FEATURES,EDGE_FEATURES
from .learning_data import read_json,sha256
from .multik_benchmark import check as check_source,verify_manifest
from .multik_core import physical_allocation
from .multik_data_core import at_target
from .multik_dataset import immutable,immutable_bytes,write_bytes,write_json
from .multik_graph import SCHEMA_VERSION,fit_normalizer,validate_graph
from .validation import family_constants

ROOT=Path(__file__).resolve().parents[1]
NUMERIC_FILES=('run_step7e.py','ris_noma/continuous_data.py','ris_noma/continuous_workers.py','ris_noma/continuous_dataset.py')
PHASES=('search','selection','audit')
SOURCE_FILES=('completion.json','plan.json','source_physics_plan.json','source_dataset_plan.json')

def inspect(source,settings,workers=2,query=None):
    source=Path(source).resolve();status_source=check_source(source)
    old=read_json(source/'plan.json');physics=read_json(source/'source_physics_plan.json')
    dataset=read_json(source/'source_dataset_plan.json');cfg=validate_settings(settings)
    if type(workers) is not int or not 1<=workers<=4:raise ValueError('Use 1 to 4 workers; 2 recommended for 32 GB RAM')
    old_ids={x for values in dataset['config']['splits'].values() for x in values}
    old_ids.update(old['config']['calibration_layout_ids']+old['config']['benchmark_layout_ids'])
    new_ids={x for values in cfg['splits'].values() for x in values}
    if new_ids & old_ids:raise ValueError('7e layouts must be outside all source 7b/7d layouts')
    if query is not None:query=validate_query(cfg,query,old_ids|new_ids)
    root_seed=physics['base_config']['root_seed'];jobs=jobs_for(cfg,root_seed,query)
    planned_counts=dict(Counter({key:0 for key in ('train','validation','test','test_unseen_k','query')}))
    for job in jobs:planned_counts[role(job)]+=len(job['cases'])
    array={k:old['config'][k] for k in ('carrier_hz','spacing_wavelengths','array_axis')}
    base={'config':cfg,'deployment':old['config']['benchmark_deployments'][0],'array_settings':array}
    for job in jobs:geometry_from(base,physics,dataset,job,cfg['search_samples'])
    circuit=physics['base_config']['n_ris']*sum(physics['base_config']['circuit_w'].values())
    if float(dbm_to_w(cfg['power_dbm_range'][0]))<=circuit:raise ValueError('Total power must exceed RIS circuit power')
    files=tuple(dict.fromkeys((*old['code_sha256'],*NUMERIC_FILES)))
    count=sum(planned_counts.values())
    return {'step':'7e','format':FORMAT,'mode':'query' if query is not None else 'dataset',
        'graph_schema':SCHEMA_VERSION,'config':cfg,'query':query,'jobs':jobs,'root_seed':root_seed,
        'streams':STREAMS,'archive_size':ARCHIVE_SIZE,'workers':workers,
        'deployment':base['deployment'],'array_settings':array,'pso_settings':old['pso_settings'],
        'planned_counts':planned_counts,'planned_cases':count,
        'max_objective_evaluations':count*len(cfg['pso_seeds'])*cfg['pso_evaluations'],
        'source_7d_verified_files':status_source['verified_files'],
        'source_sha256':{name:sha256(source/name) for name in SOURCE_FILES},
        'code_sha256':{name:sha256(ROOT/name) for name in files},
        'versions':{'system':platform.system(),'python':platform.python_version(),'numpy':np.__version__,
            'scipy':scipy.__version__,'filelock':version('filelock')},
        'source_versions':old['versions'],'power_sampling':'randomized Latin hypercube; uniform marginals in dBm, not watts',
        'parameter_sampling':'full-precision continuous P/DEP; same per-layout schedule across K; no rounding or lookup',
        'physics_policy':'unchanged scalar Active-RIS noise, hardware distortion, Bob phase alignment and average power budget',
        'search_policy':'random-initialized full PSO only; declared DEP and QoS search margins; archive up to five per seed',
        'selection_policy':'search-feasible and independent-bank accepted, then highest selection-bank Bob rate; no cross-case pooling',
        'audit_policy':'independent audit after all selections freeze; no label filtering or reselection on audit',
        'normalization_policy':'fit on exported train graphs only; test/validation not used',
        'new_weights_trained':False,'final_unseen_test':False}

def freeze(output,phase,plan):
    paths=[Path(output)/phase/(job['id']+'.json') for job in plan['jobs']]
    if any(read_group(output,phase,job) is None for job in plan['jobs']):raise ValueError('Cannot freeze incomplete stage')
    immutable(Path(output)/(phase+'_frozen.json'),{'phase':phase,'plan_sha256':sha256(Path(output)/'plan.json'),
        'sha256':{p.relative_to(output).as_posix():sha256(p) for p in paths}})

def execute_stage(output,phase,plan):
    pending=[job for job in plan['jobs'] if read_group(output,phase,job) is None]
    done=len(plan['jobs'])-len(pending);workers=plan['workers']
    print(f"7e {phase}: {done}/{len(plan['jobs'])} groups committed",flush=True)
    function={'search':search_job,'selection':select_job,'audit':audit_job}[phase]
    if workers==1:
        for job in pending:
            name=function((str(output),job));done+=1
            print(f"7e {phase} {done}/{len(plan['jobs'])}: {name}",flush=True)
    elif pending:
        todo=iter(pending);active={};executor=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'))
        def submit():
            job=next(todo,None)
            if job is not None:active[executor.submit(function,(str(output),job))]=job
        try:
            for _ in range(workers):submit()
            while active:
                finished,_=wait(active,return_when=FIRST_COMPLETED)
                for future in finished:
                    name=future.result();del active[future];done+=1
                    print(f"7e {phase} {done}/{len(plan['jobs'])}: {name}",flush=True);submit()
        finally:
            for future in active:future.cancel()
            executor.shutdown(wait=True,cancel_futures=True)
    freeze(output,phase,plan)

def collect(output):
    output=Path(output);plan=read_json(output/'plan.json');ledger=[];exports={k:[] for k in plan['planned_counts']}
    total_evaluations=0
    for job in plan['jobs']:
        search=read_group(output,'search',job);selected=read_group(output,'selection',job);audit=read_group(output,'audit',job)
        total_evaluations+=search['objective_evaluations']
        for case,choice,final in zip(search['cases'],selected['rows'],audit['rows']):
            assert case['case_id']==choice['case_id']==final['case_id']
            chosen=next((x for x in choice['candidates'] if x['candidate_id']==choice['selected_candidate_id']),None)
            row={'case_id':case['case_id'],'layout_id':job['layout_id'],'n_public':job['n_public'],
                'split':job['split'],'regime':job['regime'],'role':role(job),'power_dbm':case['power_dbm'],'xi':case['xi'],
                'graph':case['graph'],'label_available':choice['label_available'],'allocation':choice['allocation'],
                'selected_candidate_id':choice['selected_candidate_id'],'missing_reason':choice['missing_reason'],
                'selection':None if chosen is None else chosen['physical'],'audit':final['audit'],
                'audit_used_for_label_filter':False}
            ledger.append(row)
            if choice['label_available']:
                exports[role(job)].append({key:row[key] for key in ('case_id','layout_id','n_public','split','regime',
                    'power_dbm','xi','graph','allocation','selected_candidate_id')} | {'audit_used_for_label_filter':False})
    return plan,ledger,exports,total_evaluations

def summary_rows(ledger,names):
    groups={}
    for row in ledger:groups.setdefault(tuple(row[x] for x in names),[]).append(row)
    result=[]
    for key,part in sorted(groups.items(),key=lambda v:str(v[0])):
        available=[r for r in part if r['label_available']]
        passed=[r for r in part if r['audit'] is not None and r['audit']['passed']]
        result.append({**dict(zip(names,key)),'planned':len(part),'labels':len(available),'audit_passed':len(passed),
            'label_coverage':len(available)/len(part),'audit_pass_fraction_all_planned':len(passed)/len(part),
            'audit_pass_fraction_labeled':len(passed)/len(available) if available else None,
            'gated_bob_rate_all_planned':sum(r['audit']['gated_bob_rate'] for r in passed)/len(part)})
    return result

def export(output):
    output=Path(output);plan,ledger,exports,evaluations=collect(output)
    def jsonl(path,values):write_bytes(path,(''.join(json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n' for v in values)).encode('utf-8'))
    jsonl(output/'ledger.jsonl',ledger)
    for name,values in exports.items():jsonl(output/'graphs'/(name+'.jsonl'),values)
    normalizer=fit_normalizer([r['graph'] for r in exports['train']]) if exports['train'] else None
    write_json(output/'graphs/normalization.json',{'available':normalizer is not None,'parameters':normalizer,
        'fit_role':'train','training_graph_count':len(exports['train'])})
    write_json(output/'graphs/feature_schema.json',{'version':SCHEMA_VERSION,'dataset_format':FORMAT,
        'node_features':list(NODE_FEATURES),'global_features':list(GLOBAL_FEATURES),'edge_features':list(EDGE_FEATURES),
        'target':'allocation=[mb, public original IDs, s]','learner_input_keys':['graph'],
        'metadata_not_features':['case_id','layout_id','split','regime','selected_candidate_id','audit_used_for_label_filter'],
        'audit_ledger_is_not_model_input':True,'continuous_power_and_dep':True})
    by_role=summary_rows(ledger,('role',));by_k=summary_rows(ledger,('role','n_public'))
    records=[{'case_id':r['case_id'],'role':r['role'],'layout_id':r['layout_id'],'n_public':r['n_public'],
        'power_dbm':r['power_dbm'],'xi':r['xi'],'label_available':r['label_available'],
        'audit_passed':r['audit'] is not None and r['audit']['passed'],
        'bob_rate':None if r['audit'] is None else r['audit']['bob_rate'],
        'dep_lower_margin':None if r['audit'] is None else r['audit']['dep_lower']-r['xi'],
        'qos_lower_margin':None if r['audit'] is None else r['audit']['qos_lower_margin'],
        'missing_reason':r['missing_reason'],'failed_checks':[] if r['audit'] is None else r['audit']['failed_checks']} for r in ledger]
    write_table(output/'cases.csv',records);write_table(output/'coverage_by_role.csv',by_role);write_table(output/'coverage_by_k.csv',by_k)
    failures=Counter(name for r in ledger if r['audit'] is not None for name in r['audit']['failed_checks'])
    summary={'step':'7e','complete':True,'scope':plan['config']['scope'],'mode':plan['mode'],'groups':len(plan['jobs']),
        'planned_cases':len(ledger),'available_labels':sum(len(v) for v in exports.values()),
        'audit_passed':sum(r['audit'] is not None and r['audit']['passed'] for r in ledger),
        'objective_evaluations':evaluations,'max_objective_evaluations':plan['max_objective_evaluations'],
        'planned_counts':plan['planned_counts'],'exported_counts':{name:len(v) for name,v in exports.items()},
        'ready_for_training':bool(exports['train'] and exports['validation']),
        'by_role':by_role,'by_k':by_k,'audit_failure_checks':dict(failures),
        'new_weights_trained':False,'final_unseen_test':False,'audit_used_for_label_filter':False,
        'limitations':['Development pilot; no claim of GNN/MLP performance before step 7f training.',
            'Label availability and teacher audit success are not neural-network accuracy.',
            'No candidate found is not a certificate of global infeasibility.',
            'Continuous interpolation domain only; no guarantee outside declared ranges.',
            'Original scalar Active-RIS noise retained; high-SNR saturation may remain.']}
    write_json(output/'summary.json',summary)
    return summary

def run(source,settings,output,workers=2,resume=False,query=None):
    output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(output)+'.lock',timeout=0):
        plan=inspect(source,settings,workers,query)
        if output.exists() and any(output.iterdir()):
            if not resume:raise ValueError('Output is nonempty; use --resume with the same plan or a new output')
            if read_json(output/'plan.json')!=plan:raise ValueError('Changed config, code, source, workers or runtime; use a new output')
            if (output/'completion.json').exists():return check(output,source)
        output.mkdir(exist_ok=True);immutable(output/'plan.json',plan)
        for name in SOURCE_FILES:immutable_bytes(output/('source_7d_'+name if name in ('completion.json','plan.json') else name),(Path(source)/name).read_bytes())
        for name in plan['code_sha256']:immutable_bytes(output/'source_snapshot'/name,(ROOT/name).read_bytes())
        for phase in PHASES:
            execute_stage(output,phase,plan);verify_manifest(ROOT,plan['code_sha256'])
        export(output)
        for name,digest in plan['source_sha256'].items():
            if sha256(Path(source)/name)!=digest:raise ValueError('Source 7d changed during run')
        files=[p for p in sorted(output.rglob('*')) if p.is_file() and p.name!='completion.json'
            and 'figures' not in p.relative_to(output).parts and not p.name.endswith('.tmp')]
        immutable(output/'completion.json',{'sha256':{p.relative_to(output).as_posix():sha256(p) for p in files}})
        return check(output,source)

def check(output,source=None):
    output=Path(output).resolve();manifest=read_json(output/'completion.json')['sha256'];verify_manifest(output,manifest)
    plan=read_json(output/'plan.json');cfg=validate_settings(plan['config'])
    if plan['format']!=FORMAT or plan['streams']!=STREAMS or plan['archive_size']!=ARCHIVE_SIZE:raise ValueError('Wrong dataset format/policy')
    verify_manifest(ROOT,plan['code_sha256']);verify_manifest(output/'source_snapshot',plan['code_sha256'])
    source_plan=read_json(output/'source_7d_plan.json');source_data=read_json(output/'source_dataset_plan.json')
    source_physics=read_json(output/'source_physics_plan.json')
    excluded={x for ids in source_data['config']['splits'].values() for x in ids}
    excluded.update(source_plan['config']['calibration_layout_ids']+source_plan['config']['benchmark_layout_ids'])
    new_ids={x for ids in cfg['splits'].values() for x in ids}
    if new_ids & excluded:raise ValueError('Source layouts leaked into new splits')
    if plan['query'] is not None:validate_query(cfg,plan['query'],excluded|new_ids)
    if plan['root_seed']!=source_physics['base_config']['root_seed']:raise ValueError('Wrong root seed')
    if plan['deployment']!=source_plan['config']['benchmark_deployments'][0] or plan['array_settings']!={k:source_plan['config'][k] for k in ('carrier_hz','spacing_wavelengths','array_axis')}:raise ValueError('Geometry assumptions changed')
    if any(plan['code_sha256'].get(k)!=v for k,v in source_plan['code_sha256'].items()):raise ValueError('Source numerical dependencies changed')
    if jobs_for(cfg,plan['root_seed'],plan['query'])!=plan['jobs']:raise ValueError('Job/continuous parameter schedule changed')
    for phase in PHASES:
        frozen=read_json(output/(phase+'_frozen.json'))
        expected={f"{phase}/{job['id']}.json" for job in plan['jobs']}
        if set(frozen['sha256'])!=expected or frozen['plan_sha256']!=sha256(output/'plan.json'):raise ValueError('Incomplete frozen phase')
        for name,digest in frozen['sha256'].items():
            if manifest.get(name)!=digest:raise ValueError('Phase differs from completion')
    for job in plan['jobs']:
        search,selected,audit=[read_group(output,phase,job) for phase in PHASES]
        if search['job']!=job or selected['job']!=job or audit['job']!=job:raise ValueError('Wrong job record')
        if len(search['cases'])!=len(job['cases']) or len(selected['rows'])!=len(job['cases']) or len(audit['rows'])!=len(job['cases']):raise ValueError('Missing planned scenario')
        if selected['search_frozen_sha256']!=sha256(output/'search_frozen.json') or audit['selection_frozen_sha256']!=sha256(output/'selection_frozen.json'):raise ValueError('Stage chain changed')
        entropies=[];count=0
        for phase,doc in zip(PHASES,(search,selected,audit)):
            m=doc['metadata'];entropies.append(tuple(m['bank_entropy']))
            expected_seed=[plan['root_seed'],job['layout_id'],job['n_public'],STREAMS[phase]]
            expected_n=cfg[{'search':'search_samples','selection':'selection_samples','audit':'audit_samples'}[phase]]
            if m['bank_entropy']!=expected_seed or m['bank_samples']!=expected_n or m['bank_role']!=phase:raise ValueError('Bank role differs')
            if m['receiver_xy']!=search['metadata']['receiver_xy'] or m['los_sha256']!=search['metadata']['los_sha256']:raise ValueError('Geometry changed across bank roles')
        if len(set(entropies))!=3:raise ValueError('Bank role seeds overlap')
        for expected,case,choice,final in zip(job['cases'],search['cases'],selected['rows'],audit['rows']):
            for key,value in expected.items():
                if any(r[key]!=value for r in (case,choice,final)):raise ValueError('Power/DEP substituted or cross-case pooling')
            if case['case_id']!=job['id']+f"_c{expected['case_index']:03d}" or case['case_id']!=choice['case_id'] or case['case_id']!=final['case_id']:raise ValueError('Wrong case identity')
            validate_graph(case['graph'])
            g=case['graph']['global_raw']
            if g[0]!=case['xi'] or not np.isclose(g[1],np.log10(float(dbm_to_w(case['power_dbm']))),rtol=0,atol=1e-12):raise ValueError('Graph does not carry continuous power/DEP')
            if case['search_dep_target']!=case['xi']+cfg['search_dep_margin']:raise ValueError('Wrong DEP search margin')
            qos=source_physics['base_config']['requirements']['qos_public']
            if len(set(qos))!=1:raise ValueError('Current pilot requires homogeneous source QoS to extend K')
            if case['search_qos_targets']!=[qos[0]+cfg['search_qos_margin']]*job['n_public']:raise ValueError('Wrong QoS search margin')
            pools=case['pools']
            if [p['seed'] for p in pools]!=(cfg['pso_seeds'] if search['layout_eligible'] else []):raise ValueError('Missing seed run')
            original=[]
            for pool in pools:
                if pool['objective_evaluations']!=cfg['pso_evaluations'] or len(pool['candidates'])>ARCHIVE_SIZE or len(pool['history'])!=cfg['pso_evaluations']//30 or not 0<=pool['feasible_evaluations']<=cfg['pso_evaluations']:raise ValueError('Wrong PSO/archive budget')
                count+=pool['objective_evaluations'];original.extend(pool['candidates'])
                for candidate in pool['candidates']:
                    if not 1<=candidate['evaluation']<=cfg['pso_evaluations'] or candidate['seed']!=pool['seed']:raise ValueError('Wrong candidate lineage')
                    physical_allocation(candidate['allocation'],job['n_public'],source_physics['base_config']['bounds'])
            if len(choice['candidates'])!=len(original):raise ValueError('Missing independent candidate score')
            ids=[x['candidate_id'] for x in original]
            if len(ids)!=len(set(ids)):raise ValueError('Duplicate candidate identity')
            for x,y in zip(original,choice['candidates']):
                if any(y[key]!=value for key,value in x.items()):raise ValueError('Selection altered candidate')
                expected_decision=at_target(y['physical'],case['xi'])
                if any(y[key]!=value for key,value in expected_decision.items()):raise ValueError('Wrong selection decision')
                if y['physical']['constants']['pool_size']!=len(job['cases'])*len(cfg['pso_seeds'])*ARCHIVE_SIZE:raise ValueError('Selection family changed')
                if y['physical']['constants']!=family_constants(cfg['selection_samples'],len(job['cases'])*len(cfg['pso_seeds'])*ARCHIVE_SIZE,job['n_public'],cfg['alpha']/(2*len(plan['jobs']))):raise ValueError('Selection confidence constants changed')
            chosen=choose_label(choice['candidates'])
            if choice['selected_candidate_id']!=(None if chosen is None else chosen['candidate_id']):raise ValueError('Label not selected by declared rule')
            if choice['label_available']!=(chosen is not None) or choice['allocation']!=(None if chosen is None else chosen['allocation']):raise ValueError('Wrong label availability/allocation')
            for key in ('case_id','case_index','power_dbm','xi','selected_candidate_id','label_available','allocation','missing_reason'):
                if final[key]!=choice[key]:raise ValueError('Audit changed label')
            if final['audit_used_for_label_filter'] is not False:raise ValueError('Audit used to filter labels')
            if final['audit'] is not None:
                d=at_target(final['audit'],case['xi'])
                if not final['label_available'] or any(final['audit'][key]!=value for key,value in d.items()):raise ValueError('Invalid final decision')
                if final['audit']['constants']['pool_size']!=len(job['cases']):raise ValueError('Audit family changed')
                if final['audit']['constants']!=family_constants(cfg['audit_samples'],len(job['cases']),job['n_public'],cfg['alpha']/(2*len(plan['jobs']))):raise ValueError('Audit confidence constants changed')
            elif final['label_available']:raise ValueError('Available label missing final audit')
        if count!=search['objective_evaluations']:raise ValueError('Wrong group evaluation count')
    actual_plan,ledger,exports,evaluations=collect(output)
    if len(ledger)!=plan['planned_cases'] or len({x['case_id'] for x in ledger})!=len(ledger):raise ValueError('Scenario denominator changed')
    if read_lines(output/'ledger.jsonl')!=ledger:raise ValueError('Ledger differs from stages')
    for name,values in exports.items():
        if read_lines(output/'graphs'/(name+'.jsonl'))!=values:raise ValueError('Export differs or was filtered using audit')
    norm=read_json(output/'graphs/normalization.json')
    expected=fit_normalizer([r['graph'] for r in exports['train']]) if exports['train'] else None
    if norm!={'available':expected is not None,'parameters':expected,'fit_role':'train','training_graph_count':len(exports['train'])}:raise ValueError('Normalizer not train-only')
    summary=read_json(output/'summary.json')
    expected_counts={name:len(v) for name,v in exports.items()}
    if summary['planned_cases']!=len(ledger) or summary['exported_counts']!=expected_counts or summary['objective_evaluations']!=evaluations:raise ValueError('Summary count mismatch')
    if summary['by_role']!=summary_rows(ledger,('role',)) or summary['by_k']!=summary_rows(ledger,('role','n_public')):raise ValueError('Summary metrics differ')
    scalar={'step':'7e','complete':True,'scope':cfg['scope'],'mode':plan['mode'],'groups':len(plan['jobs']),
        'available_labels':sum(expected_counts.values()),'audit_passed':sum(r['audit'] is not None and r['audit']['passed'] for r in ledger),
        'planned_counts':plan['planned_counts'],'max_objective_evaluations':plan['max_objective_evaluations'],
        'ready_for_training':bool(exports['train'] and exports['validation'])}
    if any(summary[k]!=v for k,v in scalar.items()):raise ValueError('Summary scalar/scope differs')
    if plan['planned_counts']!={name:sum(role(job)==name for job in plan['jobs'] for _ in job['cases']) for name in plan['planned_counts']}:raise ValueError('Planned split counts differ')
    if plan['max_objective_evaluations']!=len(ledger)*len(cfg['pso_seeds'])*cfg['pso_evaluations']:raise ValueError('Planned PSO budget differs')
    if summary['audit_used_for_label_filter'] or summary['new_weights_trained'] or summary['final_unseen_test']:raise ValueError('Wrong scope')
    for name in SOURCE_FILES:
        copied=output/('source_7d_'+name if name in ('completion.json','plan.json') else name)
        if sha256(copied)!=plan['source_sha256'][name]:raise ValueError('Copied source changed')
    if source is not None:
        check_source(source)
        for name,digest in plan['source_sha256'].items():
            if sha256(Path(source)/name)!=digest:raise ValueError('Source 7d differs')
    return summary | {'verified_files':len(manifest)}

def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]

def status(output):
    output=Path(output);plan=read_json(output/'plan.json')
    return {'step':'7e','mode':plan['mode'],'scope':plan['config']['scope'],'groups':len(plan['jobs']),
        'planned_cases':plan['planned_cases'],'complete':(output/'completion.json').exists(),
        **{phase:sum(read_group(output,phase,job) is not None for job in plan['jobs']) for phase in PHASES}}
