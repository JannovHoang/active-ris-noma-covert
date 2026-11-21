"""7b: generate a larger variable-K PSO graph dataset; no model training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for action in ("inspect", "run"):
        command = commands.add_parser(action)
        command.add_argument("--previous", type=Path, required=True, help="Completed step7a_multik_foundation directory")
        command.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "configs/multik_dataset_7b.json")
        if action == "run":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--workers", type=int, default=1)
            command.add_argument("--resume", action="store_true")
            command.add_argument("--no-plots", action="store_true")
    for action in ("status", "check", "plot"):
        command = commands.add_parser(action)
        command.add_argument("--output", type=Path, required=True)
        if action == "check":
            command.add_argument("--replay-groups", type=int, default=0, help="Also rebuild selected bank groups and re-evaluate their saved labels")
    args = parser.parse_args()
    try:
        if args.command == "plot":
            from ris_noma.multik_dataset_plots import plot
            result = plot(args.output)
        else:
            from ris_noma import multik_dataset as data
            if args.command == "status":
                result = data.status(args.output)
            elif args.command == "check":
                result = data.check(args.output, args.replay_groups)
            else:
                settings = data.read_json(args.config)
                if args.command == "inspect":
                    plan, _ = data.inspect(args.previous, settings)
                    result = {key: plan[key] for key in ("step", "config", "planned_counts", "planned_cases", "max_pso_evaluations",
                              "source_los_policy", "geometry_policy", "selection_policy", "heldout_policy", "estimated_peak_bank_gib_per_worker")}
                    result["layout_k_groups"] = len(plan["jobs"])
                else:
                    result = data.run(args.previous, args.output, settings, args.workers, args.resume)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.command == "run" and not args.no_plots:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "plot", "--output", str(args.output)], check=True)
    except KeyboardInterrupt:
        parser.exit(130, "7b interrupted. Completed groups are retained; rerun the same command with --resume.\n")
    except subprocess.CalledProcessError:
        parser.exit(1, "7b dataset saved. Use plot to regenerate figures.\n")
    except (ValueError, OSError, ArithmeticError) as error:
        parser.exit(1, f"Step 7b: {error}\n")


if __name__ == "__main__":
    main()
