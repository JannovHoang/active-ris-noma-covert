"""7c: train shared variable-K GNN/pooled MLP and validate before frozen tests."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[name]="1"


def brief(result):
    keys=("step","complete","labeled_counts","fits","candidate_checkpoints","selected_views","model_parameters","validation_rows","test_rows","source_7b_unchanged","verified_files")
    return {key:result[key] for key in keys if key in result}|{"test_metrics":result["metrics"]["by_view"],"across_seeds":result["metrics"]["across_seeds"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    for action in ("inspect","run"):
        p=commands.add_parser(action)
        p.add_argument("--dataset",type=Path,required=True)
        p.add_argument("--config",type=Path,default=Path(__file__).resolve().parent/"configs/multik_training_7c.json")
        if action=="run":
            p.add_argument("--output",type=Path,required=True)
            p.add_argument("--workers",type=int,default=1,help="CPU processes for independent physical banks")
            p.add_argument("--resume",action="store_true")
            p.add_argument("--no-plots",action="store_true")
    for action in ("status","check","plot"):
        p=commands.add_parser(action)
        p.add_argument("--output",type=Path,required=True)
        if action=="check":p.add_argument("--dataset",type=Path)
    args=parser.parse_args()
    try:
        if args.command=="plot":
            from ris_noma.multik_learning_plots import plot
            result=plot(args.output)
        else:
            from ris_noma import multik_learning as learning
            if args.command=="check":result=brief(learning.check(args.output,args.dataset))
            elif args.command=="status":result=learning.status(args.output)
            elif args.command=="inspect":
                plan,*_=learning.inspect(args.dataset,learning.read_json(args.config))
                result={key:plan[key] for key in ("step","config","labeled_counts","training_public_counts","unseen_public_counts","model_parameters","max_validation_candidates","max_test_views","planned_validation_rows","planned_test_rows","architecture_policy","physical_selection","test_policy")}
            else:result=brief(learning.run(args.dataset,args.output,learning.read_json(args.config),args.workers,args.resume))
        print(json.dumps(result,ensure_ascii=False,indent=2))
        if args.command=="run" and not args.no_plots:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),"plot","--output",str(args.output)],check=True)
    except KeyboardInterrupt:
        parser.exit(130,"7c interrupted. Keep the output and rerun the same command with --resume.\n")
    except subprocess.CalledProcessError:
        parser.exit(1,"7c numerical results saved; use plot to regenerate figures.\n")
    except (ValueError,OSError,ArithmeticError) as error:
        parser.exit(1,f"Step 7c: {error}\n")


if __name__=="__main__":main()
