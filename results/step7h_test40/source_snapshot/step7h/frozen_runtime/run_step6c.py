"""Compare full PSO and three short PSO methods on the multi-power pilot."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "run"):
        command = sub.add_parser(name)
        command.add_argument("--dataset", required=True, help="Completed 6a output with plan.json and graphs/")
        command.add_argument("--training", required=True, help="Completed 6b output on the same dataset")
        command.add_argument("--config", default=Path(__file__).resolve().parent / "configs/benchmark_pilot_6c.json")
        if name == "run":
            command.add_argument("--output", required=True, help="Benchmark output directory")
            command.add_argument("--resume", action="store_true", help="Continue an identical run")
            command.add_argument("--no-plots", action="store_true")
    sub.add_parser("check", help="Verify the frozen search and all numerical results").add_argument("--output", required=True)
    sub.add_parser("plot", help="Plot the completed comparison").add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "plot":
            from ris_noma.expanded_benchmark_plots import plot_benchmark
            print(json.dumps(plot_benchmark(args.output), ensure_ascii=False, indent=2))
            return

        from ris_noma.expanded_benchmark import check_benchmark, inspect_benchmark, run_benchmark
        from ris_noma.learning_data import read_json

        if args.command == "inspect":
            plan, _, _ = inspect_benchmark(Path(args.dataset), Path(args.training), read_json(args.config))
            keys = ("step", "scope", "split_counts", "selected_checkpoints", "search_runs_planned", "pso_evaluations_planned")
            print(json.dumps({key: plan[key] for key in keys}, indent=2))
            return
        if args.command == "check":
            result = check_benchmark(args.output)
        else:
            result = run_benchmark(args.dataset, args.training, args.output, read_json(args.config), args.resume)
        keys = ("step", "benchmark_complete", "search_runs", "total_pso_evaluations", "audit")
        print(json.dumps({key: result[key] for key in keys}, indent=2), flush=True)
        if args.command == "run" and not args.no_plots:
            # Keep the Torch and Matplotlib runtime libraries in separate Windows processes.
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "plot", "--output", args.output], check=True)
    except subprocess.CalledProcessError:
        parser.exit(1, "Step 6c: numerical results saved. Run the plot command in the isolated environment.\n")
    except (ValueError, OSError, ArithmeticError) as error:
        parser.exit(1, f"Step 6c: {error}\n")


if __name__ == "__main__":
    main()
