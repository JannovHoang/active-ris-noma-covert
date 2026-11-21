"""Measure paired PSO performance at several budgets on new K=2 layouts."""
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
        command.add_argument("--dataset", required=True, type=Path)
        command.add_argument("--training", required=True, type=Path)
        command.add_argument("--benchmark", required=True, type=Path)
        command.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "configs/budget_6d.json")
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
            from ris_noma.budget_plots import plot_budget
            print(json.dumps(plot_budget(args.output), indent=2))
            return

        from ris_noma.budget_benchmark import check_budget, inspect_budget, run_budget
        from ris_noma.learning_data import read_json

        if args.command == "inspect":
            plan, _ = inspect_budget(args.dataset, args.training, args.benchmark, read_json(args.config), args.workers)
            keys = ("step", "layout_ids", "budgets", "primary_budget", "paired_trajectories_planned",
                    "psovals_planned", "audit_rows_planned", "selected_checkpoint")
            print(json.dumps({key: plan[key] for key in keys}, indent=2))
            return
        if args.command == "check":
            result = check_budget(args.output)
        else:
            result = run_budget(args.dataset, args.training, args.benchmark, args.output,
                                read_json(args.config), args.workers, args.resume)
        keys = ("step", "complete", "planned_layouts", "eligible_layouts", "paired_trajectories",
                "psovals", "audit_rows", "primary")
        print(json.dumps({key: result[key] for key in keys}, indent=2), flush=True)
        if args.command == "run" and not args.no_plots:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "plot", "--output", str(args.output)],
                           check=True)
    except subprocess.CalledProcessError:
        parser.exit(1, "Step 6d: numerical results saved. Use the plot command to regenerate figures.\n")
    except (ValueError, OSError, ArithmeticError) as error:
        parser.exit(1, f"Step 6d: {error}\n")


if __name__ == "__main__":
    main()
