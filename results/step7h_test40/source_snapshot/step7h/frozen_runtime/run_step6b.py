"""Run and verify the supervised GNN/MLP pilot for expanded step 6a data."""
import argparse
import json
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

from ris_noma.expanded_learning import check_output, run_pilot
from ris_noma.learning_data import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Train, freeze, score and audit the pilot")
    run.add_argument("--dataset", required=True, help="Completed 6a output containing plan.json and graphs/")
    run.add_argument("--output", required=True, help="A new directory for the 6b run")
    run.add_argument("--config", default=Path(__file__).resolve().parent / "configs/training_pilot_6b.json")
    run.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    run.add_argument("--audit-samples", type=int, default=100000)
    run.add_argument("--bank-seed", type=int, default=1)
    run.add_argument("--alpha", type=float, default=.05)
    sub.add_parser("check", help="Verify the completed run against its recorded hashes").add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "check":
            result = check_output(args.output)
        else:
            result = run_pilot(args.dataset, args.output, read_json(args.config), args.device,
                               args.audit_samples, args.bank_seed, args.alpha)
        print(json.dumps({key: result[key] for key in (
            "step", "complete", "scope", "checkpoints", "selected_by_validation", "audit",
        )}, indent=2))
    except (ValueError, OSError, ArithmeticError) as error:
        parser.exit(1, f"Step 6b: {error}\n")


if __name__ == "__main__":
    main()
