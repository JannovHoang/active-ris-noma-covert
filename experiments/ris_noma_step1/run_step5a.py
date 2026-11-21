"""Step 5a: statistical SIC graph dataset, grouped splits and pilot generation."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import run_step3b as step3b
import run_step4a as step4a
import run_step4b as step4b
from ris_noma.config import load_config
from ris_noma.pso import load_pso_settings
from ris_noma.dataset import export_dataset, check_dataset, validate_partition, NEW_CODE, ROOT


def generate(c, settings, policy, partition, output, resume=False):
    ids = [lid for role in ["train", "validation", "test"] for lid in partition["splits"][role]]
    validate_partition(partition, ids)
    seeds = partition.get("pso_seeds", [1, 2, 3])
    alpha = partition.get("study_alpha", .05)
    audit_seed = partition.get("heldout_seed_new", 2)
    if type(audit_seed) is not int or audit_seed < 0 or audit_seed == policy["heldout_seed"]:
        raise ValueError("New heldout seed must differ from source heldout seed")
    study = {"layout_ids": sorted(ids), "seeds": seeds, "study_alpha": alpha,
             "notes": "Predeclared pilot layout split; pilot test is developmental, not the reserved final GNN test"}
    output = Path(output).resolve()
    plan = {"step": "5a", "config": c, "pso_settings": asdict(settings), "policy": policy,
            "partition": partition, "study": study, "heldout_seed_new": audit_seed,
            "code_sha256": {name: step4a.sha(ROOT / name) for name in [*NEW_CODE, *step4a.TRACKED, "run_step4b.py", "ris_noma/cross_xi.py"]}}
    path = output / "dataset_plan.json"
    if output.exists() and any(output.iterdir()):
        if not resume or not path.exists() or step4a.read_json(path) != plan:
            raise ValueError("Nonempty pilot output or changed plan; use --resume with the same plan or a new output")
    else:
        output.mkdir(parents=True, exist_ok=True)
        # Partition is written before any label optimization or audit exists.
        step3b.write_json(path, plan)
    specs, _ = step3b.make_specs(c["requirements"]["xi"], dict(policy, seeds=seeds))
    budget = len(ids) * len(specs) * settings.swarm_size * settings.sweeps
    print(f"Pilot: {len(ids)} layouts | at most {len(ids)*len(c['requirements']['xi'])} labeled graphs | PSO budget if eligible={budget}", flush=True)
    step4a.run_study(deepcopy(c), settings, deepcopy(policy), study, output / "label_study4a", resume=resume)
    step4b.run(output / "label_study4a", output / "selection4b", audit_seed, resume=resume)
    status = export_dataset(output / "selection4b", output / "graphs", partition=partition, resume=resume)
    step3b.write_json(output / "pilot_status.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export", help="Export an existing step-4b study without new PSO")
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--label-source", help="Original full step-4a directory, if it was moved")
    p.add_argument("--split-config", help="Optional predeclared partition; default is development/train only")
    p.add_argument("--resume", action="store_true")
    p = sub.add_parser("generate", help="Generate a new pilot with the unchanged step-4a/4b evaluators")
    p.add_argument("--config", default="configs/legacy_physics_k2_pilot.json")
    p.add_argument("--pso-config", default="configs/pso_inherited_coefficients.json")
    p.add_argument("--validation-config", default="configs/validation_pilot.json")
    p.add_argument("--dataset-config", default="configs/dataset_pilot_10.json")
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true")
    p = sub.add_parser("check", help="Check an already exported graph dataset without modifying it")
    p.add_argument("--dataset", required=True)
    args = parser.parse_args()
    if args.command == "export":
        partition = step4a.read_json(args.split_config) if args.split_config else None
        result = export_dataset(args.source, args.output, partition, args.label_source, args.resume)
    elif args.command == "generate":
        settings, _ = load_pso_settings(args.pso_config)
        result = generate(load_config(args.config), settings, step3b.load_policy(args.validation_config),
                          step4a.read_json(args.dataset_config), args.output, args.resume)
    else:
        result = check_dataset(args.dataset)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
