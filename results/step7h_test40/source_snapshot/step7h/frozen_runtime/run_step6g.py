"""6g: frozen GNN/PSO candidate pools, independent selection, then fresh audit."""
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
        c = commands.add_parser(name)
        c.add_argument("--previous", type=Path, required=True, help="Completed Windows 6f result directory")
        c.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "configs/selection_bank_6g.json")
        c.add_argument("--workers", type=int, default=4)
        if name == "run":
            c.add_argument("--output", type=Path, required=True)
            c.add_argument("--resume", action="store_true")
            c.add_argument("--no-plots", action="store_true")
    for name in ("check", "plot"):
        commands.add_parser(name).add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "plot":
            from ris_noma.selection_bank_plots import plot_experiment
            result = plot_experiment(args.output)
        elif args.command == "check":
            from ris_noma.selection_bank import check_experiment
            result = brief(check_experiment(args.output))
        else:
            from ris_noma.learning_data import read_json
            from ris_noma.selection_bank import inspect_experiment, run_experiment
            plan, source = inspect_experiment(args.previous, read_json(args.config), args.workers)
            if args.command == "inspect":
                result = {k: plan[k] for k in ("step", "selected_model", "validation_layout_ids", "test_layout_ids",
                    "budgets", "primary_budget", "full_evaluations", "candidate_limit", "selection_samples",
                    "audit_samples", "validation_pso_evaluations_planned", "test_pso_evaluations_planned",
                    "validation_audit_rows_planned", "test_audit_rows_planned")}
            else:
                result = brief(run_experiment(args, plan, source))
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        if args.command == "run" and not args.no_plots:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "plot", "--output", str(args.output)], check=True)
    except subprocess.CalledProcessError:
        parser.exit(1, "6g numerical results saved; use plot to regenerate the figures.\n")
    except (ValueError, OSError, ArithmeticError) as error:
        parser.exit(1, f"Step 6g: {error}\n")


def brief(result):
    return {k: result[k] for k in ("step", "complete", "chosen_search_margin", "test_layouts", "test_cases",
                                   "test_audit_rows", "test_pso_evaluations", "primary")}


if __name__ == "__main__":
    main()
