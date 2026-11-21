"""7g: nested train layouts, matched large/small GNN/MLP, fixed validation, closed final test."""
import argparse
import json
import os
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[name]="1"


def add_sources(parser):
    parser.add_argument("--dataset",type=Path,required=True,help="Completed main 7e dataset")
    parser.add_argument("--training",type=Path,required=True,help="Completed main Windows 7f result")
    parser.add_argument("--config",type=Path,default=Path(__file__).resolve().parent/"configs/learning_curve_7g.json")
    parser.add_argument("--workers",type=int,default=2)


def add_stage(parser):
    parser.add_argument("--stage",type=int,required=True,help="Number of train layouts, e.g. 32, 64, 128")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("inspect");add_sources(p);add_stage(p)
    p.add_argument("--corpus",type=Path)
    for command in ("prepare","run"):
        p=sub.add_parser(command);add_sources(p);add_stage(p)
        p.add_argument("--corpus",type=Path,required=True)
        p.add_argument("--resume",action="store_true")
        if command=="run":
            p.add_argument("--output",type=Path,required=True)
            p.add_argument("--no-plots",action="store_true")
    p=sub.add_parser("train");add_stage(p)
    p.add_argument("--corpus",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--workers",type=int,default=2);p.add_argument("--resume",action="store_true")
    p.add_argument("--no-plots",action="store_true")
    p=sub.add_parser("check")
    p.add_argument("--corpus",type=Path,required=True);p.add_argument("--output",type=Path)
    p.add_argument("--stage",type=int)
    p=sub.add_parser("status")
    p.add_argument("--corpus",type=Path,required=True);p.add_argument("--output",type=Path)
    p=sub.add_parser("plot");p.add_argument("--output",type=Path,required=True)
    p=sub.add_parser("seal")
    p.add_argument("--corpus",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--reason",required=True,help="Reason to stop after the completed prefixes")
    p=sub.add_parser("diagnose")
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--training",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True,help="New folder for 7f diagnostic JSON/CSV")
    a=parser.parse_args()
    try:
        from ris_noma import curve_data as data
        from ris_noma import curve_learning as learning
        if a.command=="inspect":
            plan=data.inspect(a.dataset,a.training,data.read_json(a.config),a.workers)
            jobs=data.stage_jobs(plan,a.stage)
            existing=0
            if a.corpus is not None and (a.corpus/"plan.json").exists():
                if data.read_json(a.corpus/"plan.json")!=plan:
                    raise ValueError("Existing corpus has another plan")
                existing=sum((data.label_folder(a.corpus,j)/"completion.json").exists() for j in jobs)
            c=plan["config"]
            result={"step":"7g","scope":c["scope"],"train_layouts":a.stage,
                "validation_layouts":len(c["validation_layout_ids"]),
                "reserved_test_layouts_not_generated":len(c["reserved_test_layout_ids"]),
                "planned_train_cases":a.stage*len(c["training_public_counts"])*c["cases_per_layout"],
                "planned_validation_cases":len(c["validation_layout_ids"])*len(c["training_public_counts"])*c["cases_per_layout"],
                "label_groups":len(jobs),"already_completed_label_groups":existing,
                "additional_label_groups":len(jobs)-existing,
                "additional_max_pso_evaluations":(len(jobs)-existing)*c["cases_per_layout"]*len(c["label"]["pso_seeds"])*c["label"]["pso_evaluations"],
                "fits_this_stage":2*2*len(c["training"]["seeds"]),
                "candidate_checkpoints_this_stage":2*2*len(c["training"]["seeds"])*(len(c["training"]["snapshot_epochs"])+1),
                "model_parameters":plan["model_parameters"],"nested_sizes":c["training_sizes"],
                "source_7f_preserved":True,"final_test_opened":False}
        elif a.command in ("prepare","run"):
            if a.command=="run" and a.resume and (a.output/"completion.json").exists():
                result=learning.check(a.corpus,a.output)
            else:
                result=data.prepare(a.corpus,a.dataset,a.training,data.read_json(a.config),a.stage,a.workers,a.resume)
                if a.command=="run":
                    result=learning.train_stage(a.corpus,a.output,a.stage,a.workers,a.resume,not a.no_plots)
        elif a.command=="train":
            result=learning.train_stage(a.corpus,a.output,a.stage,a.workers,a.resume,not a.no_plots)
        elif a.command=="check":
            result=data.check_corpus(a.corpus,a.stage)
            if a.output is not None:
                result={"corpus":result,"training":learning.check(a.corpus,a.output,a.stage)}
        elif a.command=="status":
            plan=data.read_json(a.corpus/"plan.json")
            result={"step":"7g","reserved_test_generated":False,
                "completed_label_groups":sum((data.label_folder(a.corpus,j)/"completion.json").exists() for j in plan["jobs"]),
                "max_label_groups":len(plan["jobs"]),
                "ready_data_sizes":[n for n in plan["config"]["training_sizes"] if (data.stage_folder(a.corpus,n)/"completion.json").exists()],
                "completed_training_sizes":[] if a.output is None else learning.completed_stages(a.output),
                "sealed":a.output is not None and (a.output/"completion.json").exists()}
        elif a.command=="diagnose":
            data.check_7f(a.training,a.dataset)
            if a.output.exists() and any(a.output.iterdir()):
                raise ValueError("Use a new diagnostic output folder")
            from ris_noma.curve_analysis import baseline_diagnostics,table
            result,rows=baseline_diagnostics(a.training)
            data.immutable(a.output/"baseline_7f_diagnostics.json",result)
            table(a.output/"public_stream_k6.csv",rows)
        elif a.command=="seal":
            result=learning.seal(a.corpus,a.output,a.reason)
        else:
            from ris_noma.curve_analysis import plot
            result=plot(a.output)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except KeyboardInterrupt:
        parser.exit(130,"7g interrupted. Keep all outputs and resume the identical command with --resume.\n")
    except (ValueError,OSError,ArithmeticError) as error:
        parser.exit(1,f"Step 7g: {error}\n")


if __name__=="__main__":
    main()
