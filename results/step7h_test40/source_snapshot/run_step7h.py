"""7h controlled benchmark. Run with -X utf8 on Windows."""
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from pathlib import Path
import argparse
import json
import multiprocessing

import step7h
from step7h import driver
from step7h.checks import check
from step7h.analysis import plot


def main():
    parser = argparse.ArgumentParser(description="7h: six methods, validation-only budget selection, sealed independent final test; no ridge")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "pilot", "smoke", "develop"):
        item = sub.add_parser(name)
        for flag in ("corpus", "training", "diagnostics"): item.add_argument("--"+flag, required=True)
        if name != "smoke": item.add_argument("--config", default=str(Path(__file__).parent / "configs/benchmark_7h.json"))
        item.add_argument("--workers", type=int, default=2)
        if name != "inspect":
            item.add_argument("--output", required=True)
            item.add_argument("--resume", action="store_true")
    item = sub.add_parser("seal")
    item.add_argument("--output", required=True)
    item = sub.add_parser("test")
    item.add_argument("--protocol", required=True)
    item.add_argument("--output", required=True)
    item.add_argument("--workers", type=int, default=2)
    item.add_argument("--resume", action="store_true")
    for name in ("check", "plot"):
        item = sub.add_parser(name)
        item.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "inspect": result = driver.inspect(args.corpus,args.training,args.diagnostics,args.config)
    elif args.command == "seal": result = driver.seal(args.output)
    elif args.command == "test": result = driver.final_test(args.protocol,args.output,args.workers,args.resume)
    elif args.command == "check": result = check(args.output)
    elif args.command == "plot":
        status = check(args.output)
        if not status["complete"]: raise ValueError("Complete the stage before regenerating figures")
        plot(args.output)
        result = {"figures_updated":True,"output":args.output}
    elif args.command == "smoke": result = driver.smoke(args.corpus,args.training,args.diagnostics,args.output,args.workers,args.resume)
    else:
        function = driver.pilot if args.command == "pilot" else driver.develop
        result = function(args.corpus,args.training,args.diagnostics,args.output,args.config,args.workers,args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
