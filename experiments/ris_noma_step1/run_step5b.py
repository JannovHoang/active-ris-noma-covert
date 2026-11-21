"""Step 5b: inspect -> reproduce audit -> train GNN/MLP -> frozen evaluation."""
from pathlib import Path
import argparse
import json
from ris_noma.learning_data import inspect_export, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inspect", help="Validate graphs and report audit flags without training")
    p.add_argument("--dataset", required=True); p.add_argument("--output")
    p = sub.add_parser("audit", help="Reproduce failed source heldout audits using the same original bank")
    p.add_argument("--dataset", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("train", help="Fit only train; select best checkpoint with validation MSE")
    p.add_argument("--dataset", required=True); p.add_argument("--output", required=True)
    p.add_argument("--config", default="configs/training_pilot_5b.json")
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    p = sub.add_parser("evaluate", help="Evaluate all frozen checkpoints and the train-mean baseline")
    p.add_argument("--dataset", required=True); p.add_argument("--training", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("physics", help="Forward-evaluate frozen predictions and PSO references on a fresh fading bank")
    p.add_argument("--dataset", required=True); p.add_argument("--predictions", required=True); p.add_argument("--output", required=True)
    p.add_argument("--samples", type=int, default=100000); p.add_argument("--bank-seed", type=int, default=3)
    p.add_argument("--alpha", type=float, default=.05)
    p.add_argument("--splits", nargs="+", choices=["train", "validation", "test"], default=["validation", "test"])
    args = parser.parse_args()
    if args.command == "inspect":
        result = inspect_export(args.dataset)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True); write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "audit":
        from ris_noma.learning_physics import reproduce_audits
        result = reproduce_audits(args.dataset, args.output)
        print(f"Audit reproduced: {len(result['reviews'])} review record(s); output={args.output}")
    elif args.command == "train":
        try:
            from ris_noma.learning_train import train_experiment
        except ModuleNotFoundError as error:
            if error.name == "torch":
                parser.error("PyTorch missing. Install with: python -m pip install -r requirements_step5b.txt")
            raise
        result = train_experiment(args.dataset, args.output, read_json(args.config), args.device)
        print(f"Training complete: {len(result['runs'])} frozen checkpoints; test not used for selection; output={args.output}")
    elif args.command == "evaluate":
        from ris_noma.learning_train import evaluate_experiment
        result = evaluate_experiment(args.dataset, args.training, args.output)
        print(json.dumps(result['aggregate'], indent=2))
    else:
        from ris_noma.learning_physics import evaluate_physics
        result = evaluate_physics(args.dataset, args.predictions, args.output, args.samples, args.bank_seed, args.alpha, args.splits)
        print(f"Physics complete: {result['rows']} evaluations; output={args.output}")


if __name__ == "__main__":
    main()
