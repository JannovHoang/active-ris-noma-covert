"""Generate the K=2 layout/power pilot and validation-selected PSO labels."""
import argparse
import json
import os

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

from ris_noma.config import load_config
from ris_noma.expanded_data import check_run, generate
from ris_noma.expanded_scenarios import make_plan
from ris_noma.learning_data import read_json
from ris_noma.pso import load_pso_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "generate"):
        command = sub.add_parser(name)
        command.add_argument("--physics-config", default="configs/legacy_physics_k2_pilot.json")
        command.add_argument("--pso-config", default="configs/pso_inherited_coefficients.json")
        command.add_argument("--config", default="configs/dataset_pilot_6a.json")
        if name == "generate":
            command.add_argument("--output", required=True)
            command.add_argument("--resume", action="store_true")
    sub.add_parser("check").add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "check":
            result = check_run(args.output)
        else:
            settings, _ = load_pso_settings(args.pso_config)
            plan = make_plan(load_config(args.physics_config), settings, read_json(args.config))
            if args.command == "plan":
                result = {key: plan[key] for key in (
                    "purpose", "planned_cases", "max_pso_evaluations",
                    "alpha_per_power_layout_stage", "experiment", "layouts",
                )}
            else:
                result = generate(plan, args.output, args.resume)
        print(json.dumps(result, ensure_ascii=True, indent=2))
    except (ValueError, OSError) as error:
        parser.exit(1, f"Step 6a: {error}\n")


if __name__ == "__main__":
    main()
