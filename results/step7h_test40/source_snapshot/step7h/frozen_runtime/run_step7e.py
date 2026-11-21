"""7e: continuous-power/DEP PSO labels and independent physical audit."""
import argparse
import json
from ris_noma.learning_data import read_json
from ris_noma import continuous_dataset as dataset

def brief(result):
    keys=('step','complete','scope','mode','groups','planned_cases','available_labels','audit_passed',
          'objective_evaluations','max_objective_evaluations','planned_counts','exported_counts','ready_for_training','verified_files')
    return {key:result[key] for key in keys if key in result}

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for command in ('inspect','run','solve'):
        p=sub.add_parser(command);p.add_argument('--source',required=True)
        p.add_argument('--config',default='configs/continuous_pilot_7e.json');p.add_argument('--workers',type=int,default=2)
        if command!='inspect':p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true')
        if command=='solve':
            p.add_argument('--power-dbm',type=float,required=True);p.add_argument('--dep',type=float,required=True)
            p.add_argument('--layout-id',type=int,required=True);p.add_argument('--public-users',type=int,required=True)
    for command in ('check','status','plot'):
        p=sub.add_parser(command);p.add_argument('--output',required=True)
        if command=='check':p.add_argument('--source')
    a=parser.parse_args()
    if a.command=='inspect':
        p=dataset.inspect(a.source,read_json(a.config),a.workers)
        result={'step':'7e','scope':p['config']['scope'],'mode':p['mode'],'groups':len(p['jobs']),
            'planned_cases':p['planned_cases'],'planned_counts':p['planned_counts'],
            'max_objective_evaluations':p['max_objective_evaluations'],
            'power_dbm_range':p['config']['power_dbm_range'],'dep_range':p['config']['dep_range'],
            'search_dep_margin':p['config']['search_dep_margin'],'search_qos_margin':p['config']['search_qos_margin']}
    elif a.command in ('run','solve'):
        query=None if a.command=='run' else {'layout_id':a.layout_id,'n_public':a.public_users,'power_dbm':a.power_dbm,'xi':a.dep}
        result=brief(dataset.run(a.source,read_json(a.config),a.output,a.workers,a.resume,query))
        from ris_noma.continuous_plots import plot
        plot(a.output)
    elif a.command=='check':result=brief(dataset.check(a.output,a.source))
    elif a.command=='status':result=dataset.status(a.output)
    else:
        from ris_noma.continuous_plots import plot
        result=plot(a.output)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
