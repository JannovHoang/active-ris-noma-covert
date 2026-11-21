"""7f data boundary: grouped continuous cases, train-only normalization, no audit features."""
from copy import deepcopy
from pathlib import Path
import numpy as np

from .continuous_data import FORMAT as DATASET_FORMAT, role
from .continuous_dataset import check as check_dataset, read_lines
from .learning_data import read_json, sha256
from .multik_core import physical_allocation
from .multik_graph import fit_normalizer, validate_graph
from .multik_learning import validate_config as validate_training_config

FORMAT='continuous_learning_7f_v1'
KINDS=('gnn','mlp')
RULES=('weighted_loss','physical')
ROLES=('train','validation','test','test_unseen_k')
SOURCE_FILES=('completion.json','plan.json','source_physics_plan.json','source_dataset_plan.json',
    'graphs/feature_schema.json','graphs/normalization.json',
    'graphs/train.jsonl','graphs/validation.jsonl','graphs/test.jsonl','graphs/test_unseen_k.jsonl')
COPIES={'plan.json':'source_7e_plan.json','completion.json':'source_7e_completion.json',
    'source_physics_plan.json':'source_physics_plan.json','source_dataset_plan.json':'source_geometry_plan.json',
    'graphs/feature_schema.json':'source_feature_schema.json','graphs/normalization.json':'source_normalization.json'}


def validate_config(settings):
    c=deepcopy(settings)
    if c.pop('scope',None) not in ('smoke','development_pilot'):
        raise ValueError('7f requires scope=smoke or development_pilot')
    checked=validate_training_config(c)
    return dict(checked,scope=settings['scope'])


def dataset_info(dataset):
    dataset=Path(dataset).resolve();status=check_dataset(dataset)
    plan=read_json(dataset/'plan.json')
    if plan['format']!=DATASET_FORMAT or plan['mode']!='dataset':
        raise ValueError('7f requires a completed 7e dataset, not a single query or 7b data')
    if not status['ready_for_training']:
        raise ValueError('7e has no nonempty train and validation labels')
    return status,plan,read_json(dataset/'source_physics_plan.json')


def load_cases(dataset,name,plan):
    """All planned cases, joined to labels if available; no ledger/audit input.

    Missing teacher labels are retained for physical evaluation. Only labeled
    rows are used for supervised loss. Original public IDs/order are preserved.
    """
    if name not in ROLES:raise ValueError('Unknown 7f input role')
    dataset=Path(dataset);exported=read_lines(dataset/'graphs'/(name+'.jsonl'))
    labels={r['case_id']:r for r in exported}
    if len(labels)!=len(exported):raise ValueError('Duplicate exported teacher labels')
    rows=[]
    for job in plan['jobs']:
        if role(job)!=name:continue
        search=read_json(dataset/'search'/(job['id']+'.json'))
        for case in search['cases']:
            label=labels.pop(case['case_id'],None)
            if validate_graph(case['graph'])!=job['n_public']:raise ValueError('Graph K differs')
            if label is not None:
                if label['audit_used_for_label_filter'] is not False:raise ValueError('Audit-filtered teacher data')
                for field,value in {'case_id':case['case_id'],'graph':case['graph'],
                    'layout_id':job['layout_id'],'n_public':job['n_public'],'split':job['split'],
                    'regime':job['regime'],'power_dbm':case['power_dbm'],'xi':case['xi']}.items():
                    if label[field]!=value:raise ValueError('Teacher case/continuous input differs')
            rows.append({'case_id':case['case_id'],'layout_id':job['layout_id'],'n_public':job['n_public'],
                'split':job['split'],'regime':job['regime'],'role':name,
                'power_dbm':case['power_dbm'],'xi':case['xi'],'graph':case['graph'],
                'order':search['order'],'allocation':None if label is None else label['allocation'],
                'has_label':label is not None})
    if labels:raise ValueError('Exported label has no planned search case')
    if len(rows)!=plan['planned_counts'][name] or len({r['case_id'] for r in rows})!=len(rows):
        raise ValueError('Planned scenario denominator differs')
    return rows


def labeled(rows):return [row for row in rows if row['has_label']]


def validate_rows(rows,name,source_plan,bounds):
    if name not in ROLES:raise ValueError('Unknown split')
    split='test' if name=='test_unseen_k' else name
    ks=source_plan['config']['unseen_public_counts' if name=='test_unseen_k' else 'training_public_counts']
    ids=set()
    for row in rows:
        if (row['role']!=name or row['split']!=split or row['regime']!=('unseen_k' if name=='test_unseen_k' else 'seen_k')
                or row['layout_id'] not in source_plan['config']['splits'][split]
                or row['n_public'] not in ks or row['case_id'] in ids):raise ValueError('Split/layout/K leakage or duplicate case')
        ids.add(row['case_id']);validate_graph(row['graph'])
        if row['has_label']!=(row['allocation'] is not None):raise ValueError('Wrong label availability')
        if row['has_label']:physical_allocation(row['allocation'],row['n_public'],bounds)
        if row['order']!=np.argsort(np.asarray(row['graph']['node_raw'])[1:,9]).tolist():
            raise ValueError('Graph SIC order differs from physical evaluator order')
        if any(name in row for name in ('audit','selection','heldout_passed')):
            raise ValueError('Physical audit outcomes must not enter learning records')


def train_normalizer(rows,exported):
    training=labeled(rows)
    value=fit_normalizer([r['graph'] for r in training])
    if (exported.get('fit_role')!='train' or exported.get('training_graph_count')!=len(training)
            or exported.get('parameters')!=value or exported.get('available') is not True):
        raise ValueError('7e normalizer differs from train-only labeled input graphs')
    return value


def source_hashes(dataset):return {name:sha256(Path(dataset)/name) for name in SOURCE_FILES}


def group_path(output,phase,stage,job):return Path(output)/f'{phase}_{stage}'/(job['id']+'.json')
