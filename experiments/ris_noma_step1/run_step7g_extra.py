"""Independent supplement; preserves the original 7g source and results."""
from pathlib import Path
import argparse
import json
import multiprocessing

import step7g_extra
from step7g_extra import driver


def main():
    parser=argparse.ArgumentParser(description="7g/32 diagnostics, scalar baseline, delta and small 8/16/32 curve; no PSO or final test")
    sub=parser.add_subparsers(dest="command",required=True)
    for command in ("inspect","analyze","pilot","fit","run"):
        p=sub.add_parser(command)
        p.add_argument("--corpus",required=True);p.add_argument("--training",required=True)
        p.add_argument("--config",default=str(Path(__file__).parent/"configs/diagnostics_7g32.json"))
        p.add_argument("--workers",type=int,default=2)
        if command!="inspect":p.add_argument("--output",required=True);p.add_argument("--resume",action="store_true")
    p=sub.add_parser("check");p.add_argument("--output",required=True);p.add_argument("--corpus");p.add_argument("--training")
    p=sub.add_parser("plot");p.add_argument("--output",required=True)
    a=parser.parse_args()
    if a.command=="inspect":result=driver.inspect(a.corpus,a.training,a.config)
    elif a.command=="check":result=driver.check(a.output,a.corpus,a.training)
    elif a.command=="plot":driver.plot(a.output);result={"figures_updated":True}
    else:result=driver.execute(a.command,a.corpus,a.training,a.output,a.config,a.workers,a.resume)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=="__main__":
    multiprocessing.freeze_support()
    main()
