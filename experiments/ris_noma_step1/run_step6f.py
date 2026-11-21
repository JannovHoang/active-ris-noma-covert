"""Step 6f: controlled loss/checkpoint ablation on frozen 6e labels."""
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
    for name in ("inspect", "run"):
        command = commands.add_parser(name)
        command.add_argument("--dataset", required=True, type=Path, help="Completed 6a dataset")
        command.add_argument("--previous", required=True, type=Path, help="Completed 6e output")
        command.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "configs/training_physics_6f.json")
        command.add_argument("--workers", type=int, default=4)
        if name == "run":
            command.add_argument("--output", required=True, type=Path)
            command.add_argument("--resume", action="store_true")
            command.add_argument("--no-plots", action="store_true")
    for name in ("check", "plot"):
        commands.add_parser(name).add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "plot":
            from ris_noma.physics_training_plots import plot_experiment
            result = plot_experiment(args.output)
        elif args.command == "check":
            from ris_noma.physics_training import check_experiment
            result = brief(check_experiment(args.output))
        else:
            from ris_noma.learning_data import read_json
            from ris_noma.physics_training import inspect_experiment, run_experiment
            plan, source = inspect_experiment(args.dataset, args.previous, read_json(args.config), args.workers)
            if args.command == "inspect":
                result = {key: plan[key] for key in ("step", "label_slack", "matched_cases", "loss_weights",
                          "selection_groups", "snapshot_epochs", "test_layout_ids", "benchmark_budget",
                          "max_validation_candidates", "test_audit_rows_planned")}
            else:
                result = brief(run_experiment(args, plan, source))
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        if args.command == "run" and not args.no_plots:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "plot", "--output", str(args.output)], check=True)
    except subprocess.CalledProcessError:
        parser.exit(1, "Step 6f numerical results saved; rerun the plot command to generate figures.\n")
    except (ValueError, OSError, ArithmeticError) as error:
        parser.exit(1, f"Step 6f: {error}\n")


def brief(result):
    return {key: result[key] for key in ("step", "complete", "label_slack", "matched_cases", "chosen_group",
                                         "test_layouts", "test_cases", "test_audit_rows", "primary")}


if __name__ == "__main__":
    main()
