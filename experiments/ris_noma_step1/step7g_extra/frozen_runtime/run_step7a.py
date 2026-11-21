"""7a: variable-K PSO/GNN foundation with a small reproducible mixed-K demo."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for action in ("inspect", "run"):
        command = commands.add_parser(action)
        command.add_argument("--previous", type=Path, required=True, help="Completed step6g_selection directory")
        command.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "configs/multik_foundation_7a.json")
        if action == "run":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--resume", action="store_true")
            command.add_argument("--no-plots", action="store_true")
    for action in ("check", "plot"):
        commands.add_parser(action).add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "plot":
            from ris_noma.multik_plots import plot
            result = plot(args.output)
        elif args.command == "check":
            from ris_noma.multik_demo import check
            result = brief(check(args.output))
        else:
            from ris_noma.learning_data import read_json
            from ris_noma.multik_demo import inspect, run
            config = read_json(args.config)
            if args.command == "inspect":
                plan, _ = inspect(args.previous, config)
                result = {"step": "7a", "purpose": plan["purpose"], "config": plan["config"],
                          "feature_dimensions": {"node": 15, "global": 24, "edge": 2},
                          "architecture_has_fixed_K_limit": False, "final_paper_benchmark": False}
            else:
                result = brief(run(args.previous, args.output, config, args.resume))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.command == "run" and not args.no_plots:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "plot", "--output", str(args.output)], check=True)
    except subprocess.CalledProcessError:
        parser.exit(1, "7a numerical results saved; use plot to regenerate figures.\n")
    except (ValueError, OSError, ArithmeticError) as error:
        parser.exit(1, f"Step 7a: {error}\n")


def brief(result):
    return {k: result[k] for k in ("step", "complete", "foundation_only", "planned_cases", "pso_evaluations", "compatibility", "shared_training", "coverage", "smoke_physics")}


if __name__ == "__main__":
    main()
