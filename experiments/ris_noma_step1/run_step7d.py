"""7d geometry-LoS development calibration and fair PSO initialization controls."""
import argparse
import json
from pathlib import Path

from ris_noma.learning_data import read_json
from ris_noma import multik_benchmark as benchmark


def brief(result):
    keys = ("step", "complete", "scope", "groups", "search_pools", "audit_rows", "benchmark_rows", "objective_evaluations", "verified_files")
    return {k: result[k] for k in keys if k in result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "run"):
        p = sub.add_parser(command)
        p.add_argument("--dataset", required=True); p.add_argument("--training", required=True)
        p.add_argument("--config", default="configs/multik_benchmark_7d.json")
        p.add_argument("--workers", type=int, default=2)
        p.add_argument("--reuse-search", help="Optional exact frozen search cache with verified source snapshots; selection/audit are always recomputed")
        if command == "run":
            p.add_argument("--output", required=True); p.add_argument("--resume", action="store_true")
    for command in ("check", "status", "plot"):
        p = sub.add_parser(command); p.add_argument("--output", required=True)
        if command == "check":
            p.add_argument("--dataset"); p.add_argument("--training")
    args = parser.parse_args()
    if args.command == "inspect":
        result = benchmark.inspect(args.dataset, args.training, read_json(args.config), args.workers, args.reuse_search)
        result = {key: result[key] for key in ("step", "config", "planned_search_pools", "planned_audit_rows", "planned_objective_evaluations", "profile_policy", "timing_policy")} | {"groups": len(result["jobs"])}
    elif args.command == "run":
        result = benchmark.run(args.dataset, args.training, args.output, read_json(args.config), args.workers, args.resume, args.reuse_search)
        from ris_noma.multik_benchmark_plots import plot
        plot(args.output)
        result = brief(result)
    elif args.command == "check": result = brief(benchmark.check(args.output, args.dataset, args.training))
    elif args.command == "status": result = benchmark.status(args.output)
    else:
        from ris_noma.multik_benchmark_plots import plot
        result = plot(args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
