"""Controlled comparison of standard and DEP-slack PSO labels for GNN training."""
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
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "labels", "run"):
        command = commands.add_parser(name)
        command.add_argument("--dataset", required=True, type=Path, help="Completed 6a output")
        command.add_argument("--training", required=True, type=Path, help="Completed 6b output")
        command.add_argument("--benchmark", required=True, type=Path, help="Completed 6d output")
        command.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "configs/robust_labels_6e.json")
        command.add_argument("--workers", type=int, default=4)
        if name != "inspect":
            command.add_argument("--output", required=True, type=Path)
            command.add_argument("--resume", action="store_true")
        if name == "run":
            command.add_argument("--no-plots", action="store_true")
    for name in ("check", "plot"):
        commands.add_parser(name).add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        if args.command == "plot":
            from ris_noma.dep_label_plots import plot_experiment
            result = plot_experiment(args.output)
        elif args.command == "check":
            from ris_noma.dep_label_learning import check_experiment
            result = brief(check_experiment(args.output))
        else:
            from ris_noma.dep_label_experiment import inspect_experiment, run_experiment
            from ris_noma.learning_data import read_json
            plan, source = inspect_experiment(args.dataset, args.training, args.benchmark, read_json(args.config), args.workers)
            if args.command == "inspect":
                result = {k: plan[k] for k in ("step", "splits", "dep_slacks", "label_evaluations_planned", "benchmark_budget")}
            else:
                result = run_experiment(args, plan, source)
                if args.command == "run":
                    result = brief(result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        if args.command == "run" and not args.no_plots:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "plot", "--output", str(args.output)], check=True)
    except subprocess.CalledProcessError:
        parser.exit(1, "Step 6e numerical results saved. Use the plot command to regenerate figures.\n")
    except (ValueError, OSError, ArithmeticError) as error:
        parser.exit(1, f"Step 6e: {error}\n")


def brief(result):
    return {key: result[key] for key in ("step", "complete", "chosen_slack", "matched_labels",
                                        "test_layouts", "test_cases", "test_audit_rows", "primary")}


if __name__ == "__main__":
    main()
