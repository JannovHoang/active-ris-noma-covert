"""PSO baseline, short PSO, and frozen GNN/MLP initialized short PSO."""
import os
for _key in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(_key,"1")
from pathlib import Path
import argparse
import json
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    for name in ["inspect","run"]:
        p=sub.add_parser(name)
        p.add_argument("--dataset",required=True);p.add_argument("--training",required=True)
        p.add_argument("--config",default="configs/benchmark_pilot_5c.json")
        if name=="run":
            p.add_argument("--output",required=True);p.add_argument("--resume",action="store_true");p.add_argument("--no-plots",action="store_true")
    p=sub.add_parser("plot",help="Render comparisons in a fresh process without importing Torch")
    p.add_argument("--output",required=True)
    p=sub.add_parser("check",help="Check committed records and frozen search integrity")
    p.add_argument("--output",required=True)
    args=parser.parse_args()
    if args.command=="plot":
        from ris_noma.benchmark_plots import plot_benchmark
        print(json.dumps(plot_benchmark(args.output),ensure_ascii=False,indent=2));return
    if args.command=="check":
        from ris_noma.benchmark_plots import read_results
        searches,audits=read_results(args.output)
        print(f"Committed records OK: {len(searches)} searches, {len(audits)} audits");return
    from ris_noma.benchmark_pso import inspect_benchmark,run_benchmark
    config=json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    if args.command=="inspect":
        _,_,info,_,_=inspect_benchmark(args.dataset,args.training,config)
        print(json.dumps(info,ensure_ascii=False,indent=2));return
    summary=run_benchmark(args.dataset,args.training,args.output,config,args.resume)
    print("Benchmark complete:",summary["search_runs"],"runs; PSO evaluations:",summary["total_pso_evaluations"],flush=True)
    if not args.no_plots:
        # A new process prevents the training/runtime DLLs from sharing a
        # process with Matplotlib when users have mixed Windows libraries.
        completed=subprocess.run([sys.executable,str(Path(__file__).resolve()),"plot","--output",str(Path(args.output).resolve())])
        if completed.returncode:
            parser.exit(1,"Numerical results saved; plotting failed. Run the plot command in the isolated project environment.\n")


if __name__=="__main__":main()
