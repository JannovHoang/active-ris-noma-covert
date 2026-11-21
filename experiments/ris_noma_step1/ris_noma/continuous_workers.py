"""Frozen PSO pools, independent label selection, and non-filtering final audit."""
from pathlib import Path
import numpy as np
from .channels import public_order
from .continuous_data import ARCHIVE_SIZE, MarginObjective, physical_context, pso_rng
from .learning_data import read_json, sha256
from .multik_benchmark_core import search_pool
from .multik_data_core import at_target
from .multik_dataset import write_json,receipt_path,read_committed
from .multik_graph import statistical_graph
from .pso import PSOSettings
from .stable_assessment import raw_assessment

def commit(path,value):
    write_json(path,value);write_json(receipt_path(path),{'sha256':sha256(path)})

def read_group(output,phase,job):
    result=read_committed(Path(output)/phase/(job['id']+'.json'))
    if result is not None and result.get('plan_sha256')!=sha256(Path(output)/'plan.json'):
        raise ValueError('Group belongs to a different 7e plan')
    return result

# Tạo kho ứng viên PSO cho một layout/K. Bank search được dùng để tối ưu;
# chưa truy cập fading selection hoặc audit.
def search_job(arguments):
    output,job=arguments;output=Path(output);plan=read_json(output/'plan.json');cfg=plan['config']
    path=output/'search'/(job['id']+'.json')
    if read_group(output,'search',job) is not None:return job['id']
    # Xác định thứ tự SIC từ bank đầu vào và giữ nguyên khi chọn/kiểm định nhãn.
    c,layout,bank,meta=physical_context(plan,output,job,'search');order=public_order(bank)
    # Ngân sách gồm bầy khởi tạo: 30 hạt * sweeps = số lần chấm đã khai báo.
    settings=PSOSettings(**dict(plan['pso_settings'],swarm_size=30,sweeps=cfg['pso_evaluations']//30))
    cases=[];total=0;eligible=None
    for case in job['cases']:
        physical=dict(c,total_dbm=case['power_dbm'])
        # Biên tìm kiếm QoS và DEP giúp tránh nghiệm sát biên; mục tiêu kiểm định
        # cuối vẫn là QoS/DEP gốc của case.
        objective=MarginObjective(physical,layout,bank,order,cfg['search_qos_margin'])
        if eligible is None:eligible=objective.layout_eligible
        pools=[]
        if eligible:
            # Nhiều seed PSO tạo các lần tìm kiếm khác nhau trên cùng scenario/bank.
            for seed in cfg['pso_seeds']:
                pool=search_pool(objective,physical,order,case['xi']+cfg['search_dep_margin'],settings,
                    pso_rng(c['root_seed'],job,seed,case['case_index']))
                assert len(pool['candidates'])<=ARCHIVE_SIZE
                pool['seed']=seed
                for candidate in pool['candidates']:
                    candidate['candidate_id']=f"s{seed}:"+candidate['candidate_id']
                    candidate['seed']=seed
                pool['raw_candidate_id']=f"s{seed}:"+pool['raw_candidate_id']
                pools.append(pool);total+=pool['objective_evaluations']
        cases.append({**case,'case_id':job['id']+f"_c{case['case_index']:03d}",
            'graph':statistical_graph(physical,layout,bank,order,case['xi']),
            'search_dep_target':case['xi']+cfg['search_dep_margin'],
            'search_qos_targets':objective.qos.tolist(),'pools':pools})
    commit(path,{'job':job,'cases':cases,'metadata':meta,'order':order.tolist(),
        'layout_eligible':bool(eligible),'objective_evaluations':total,'plan_sha256':sha256(output/'plan.json')})
    return job['id']

# Chỉ lấy ứng viên vừa khả thi ở search vừa qua selection độc lập.
# Trong tập đó ưu tiên rate Bob cao nhất; không bảo đảm tối ưu toàn cục.
def choose_label(candidates):
    valid=[x for x in candidates if x['search_feasible'] and x['passed']]
    return min(valid,key=lambda x:(-x['physical']['bob_rate'],x['candidate_id'])) if valid else None

# Chấm kho đã cố định trên bank selection mới. Toàn bộ search phải hoàn tất
# trước khi bank này được phép sinh, tránh tối ưu ngược theo validation.
def select_job(arguments):
    output,job=arguments;output=Path(output)
    if not (output/'search_frozen.json').exists():raise ValueError('Freeze every search before selection bank generation')
    if read_group(output,'selection',job) is not None:return job['id']
    plan=read_json(output/'plan.json');cfg=plan['config'];search=read_group(output,'search',job)
    c,layout,bank,meta=physical_context(plan,output,job,'selection');order=np.asarray(search['order'],int)
    # Họ kiểm định bao phủ mọi case, seed và vị trí trong kho ứng viên;
    # không chỉ tính ứng viên cuối cùng đã chọn.
    family=len(job['cases'])*len(cfg['pso_seeds'])*ARCHIVE_SIZE
    alpha=cfg['alpha']/(2*len(plan['jobs']));rows=[]
    for case in search['cases']:
        candidates=[]
        for pool in case['pools']:
            for candidate in pool['candidates']:
                physical=raw_assessment(dict(c,total_dbm=case['power_dbm']),layout,bank,order,
                    candidate['allocation'],family,alpha)
                candidates.append({**candidate,'physical':physical,**at_target(physical,case['xi'])})
        chosen=choose_label(candidates)
        reason=None if chosen else ('bob_mean_gain_rule_failed' if not search['layout_eligible'] else
            'no_search_feasible_candidate' if not any(x['search_feasible'] for x in candidates) else 'no_candidate_passed_selection')
        rows.append({key:case[key] for key in ('case_id','case_index','power_dbm','xi')} | {
            'candidates':candidates,'selected_candidate_id':None if chosen is None else chosen['candidate_id'],
            'label_available':chosen is not None,'allocation':None if chosen is None else chosen['allocation'],
            'missing_reason':reason})
    commit(output/'selection'/(job['id']+'.json'),{'job':job,'rows':rows,'metadata':meta,
        'search_frozen_sha256':sha256(output/'search_frozen.json'),'plan_sha256':sha256(output/'plan.json')})
    return job['id']

# Dùng bank thứ ba để báo cáo nhãn đã chọn. Audit không đổi nhãn,
# không chọn lại và không lọc bỏ các nhãn trượt để làm đẹp dữ liệu.
def audit_job(arguments):
    output,job=arguments;output=Path(output)
    if not (output/'selection_frozen.json').exists():raise ValueError('Freeze every selection before final audit bank generation')
    if read_group(output,'audit',job) is not None:return job['id']
    plan=read_json(output/'plan.json');cfg=plan['config'];selected=read_group(output,'selection',job)
    search=read_group(output,'search',job);order=np.asarray(search['order'],int)
    c,layout,bank,meta=physical_context(plan,output,job,'audit');rows=[]
    family=len(job['cases']);alpha=cfg['alpha']/(2*len(plan['jobs']))
    for chosen in selected['rows']:
        audit=None
        if chosen['label_available']:
            raw=raw_assessment(dict(c,total_dbm=chosen['power_dbm']),layout,bank,order,chosen['allocation'],family,alpha)
            audit={**raw,**at_target(raw,chosen['xi'])}
        rows.append({key:chosen[key] for key in ('case_id','case_index','power_dbm','xi','selected_candidate_id',
            'label_available','allocation','missing_reason')} | {'audit':audit,'audit_used_for_label_filter':False})
    commit(output/'audit'/(job['id']+'.json'),{'job':job,'rows':rows,'metadata':meta,
        'selection_frozen_sha256':sha256(output/'selection_frozen.json'),'plan_sha256':sha256(output/'plan.json')})
    return job['id']
