"""7f: train variable-K GNN/pooled MLP on continuous 7e inputs, with frozen physical tests."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'


def brief(result):
    keys=('step','complete','scope','labeled_counts','planned_counts','fits','candidate_checkpoints','selected_views',
          'model_parameters','validation_rows','test_rows','source_7e_unchanged','verified_files')
    return {key:result[key] for key in keys if key in result}|{'test_across_seeds':result['metrics']['across_seeds']}


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for command in ('inspect','run'):
        p=sub.add_parser(command);p.add_argument('--dataset',type=Path,required=True)
        p.add_argument('--config',type=Path,default=Path(__file__).resolve().parent/'configs/continuous_training_7f.json')
        p.add_argument('--workers',type=int,default=2);p.add_argument('--device',choices=['cpu'],default='cpu')
        if command=='run':
            p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--no-plots',action='store_true')
    for command in ('check','status','plot'):
        p=sub.add_parser(command);p.add_argument('--output',type=Path,required=True)
        if command=='check':p.add_argument('--dataset',type=Path)
    a=parser.parse_args()
    try:
        if a.command=='plot':
            from ris_noma.continuous_learning_plots import plot
            result=plot(a.output)
        else:
            from ris_noma import continuous_learning as learning
            if a.command=='inspect':
                plan,*_=learning.inspect(a.dataset,learning.read_json(a.config),a.workers)
                keys=('step','config','workers','labeled_counts','planned_counts','model_parameters',
                      'max_validation_candidates','max_test_views','planned_validation_rows','planned_test_rows',
                      'architecture_policy','physical_policy','test_policy')
                result={key:plan[key] for key in keys}
            elif a.command=='run':result=brief(learning.run(a.dataset,a.output,learning.read_json(a.config),a.workers,a.resume))
            elif a.command=='check':result=brief(learning.check(a.output,a.dataset))
            else:result=learning.status(a.output)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        if a.command=='run' and not a.no_plots:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'plot','--output',str(a.output)],check=True)
    except KeyboardInterrupt:parser.exit(130,'7f interrupted; keep output and resume the identical command with --resume.\n')
    except subprocess.CalledProcessError:parser.exit(1,'7f numerical results saved; run plot again to regenerate figures.\n')
    except (ValueError,OSError,ArithmeticError) as error:parser.exit(1,f'Step 7f: {error}\n')


if __name__=='__main__':main()
