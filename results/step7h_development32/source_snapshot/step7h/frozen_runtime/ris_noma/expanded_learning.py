"""Step 6b supervised pilot on the grouped, multi-power export from step 6a."""
from copy import deepcopy
from pathlib import Path
import platform

import numpy as np
import scipy
import torch

from .expanded_data import check_run, save_json
from .expanded_scenarios import FORMAT, ROOT
from .expanded_learning_physics import AUDIT_STREAM, CASE_FIELDS, PREDICTION_FIELDS, audit_predictions
from .graph_data import EDGE_FEATURES, GLOBAL_FEATURES, NODE_FEATURES, SCHEMA_VERSION
from .learning_data import INPUT_KEYS, SPLITS, TARGET_KEY, file_fingerprints, load_split, read_json, sha256
from .learning_train import configure_runtime, load_checkpoint, predict, regression_metrics, save_csv, train_one
from .learning_train import validate_training_config


CODE_FILES = (
    "run_step6b.py", "ris_noma/expanded_learning.py", "ris_noma/expanded_learning_physics.py",
    "ris_noma/learning_train.py", "ris_noma/learning_models.py", "ris_noma/learning_physics.py",
    "ris_noma/learning_data.py", "ris_noma/graph_data.py", "ris_noma/expanded_data.py",
    "ris_noma/expanded_scenarios.py", "ris_noma/channels.py", "ris_noma/evaluator.py",
    "ris_noma/metrics.py", "ris_noma/dep.py", "ris_noma/validation.py", "ris_noma/objective.py",
)


def validate_source(source):
    status = check_run(source)
    if status["step"] != "6a" or not status["complete"] or not status["all_splits_have_labels"]:
        raise ValueError("Need a completed step 6a run with labels in every split")
    graphs = source / "graphs"
    schema = read_json(graphs / "feature_schema.json")
    expected = {
        "version": SCHEMA_VERSION, "dataset_format": FORMAT, "case_key": list(CASE_FIELDS),
        "model_inputs": list(INPUT_KEYS), "node_features": NODE_FEATURES,
        "global_features": GLOBAL_FEATURES, "edge_features": EDGE_FEATURES,
        "node_order": ["Bob", "public_original_0", "public_original_1"],
        "target_order": ["mb", "public_original_0", "public_original_1", "s"],
    }
    if any(schema.get(key) != value for key, value in expected.items()):
        raise ValueError("Step 6a graph schema does not match the 6b models")
    return status, schema


def save_predictions(output, splits, training_status, plan):
    """Generate predictions only after all validation-selected checkpoints froze."""
    folder = output / "predictions"
    folder.mkdir()
    configure_runtime(0, plan["config"]["threads"], "cpu")
    averages = splits["train"][TARGET_KEY].mean(axis=0)
    variants = [("train_mean", None, None)] + [
        (run["name"], run["model"], run["seed"]) for run in training_status["runs"]
    ]
    hashes, rows = {}, []
    for name, kind, seed in variants:
        model = None
        if kind is not None:
            path = output / "training" / name / "best_model.pt"
            if sha256(path) != training_status["checkpoint_sha256"][name]:
                raise ValueError(f"Frozen checkpoint changed: {name}")
            model, checkpoint = load_checkpoint(path)
            if checkpoint["dataset_sha256"] != plan["dataset_sha256"]:
                raise ValueError(f"Checkpoint source mismatch: {name}")
        for role, data in splits.items():
            labels = data[TARGET_KEY]
            y = np.broadcast_to(averages, labels.shape).copy() if model is None else predict(model, data)
            if y.shape != labels.shape or not np.all(np.isfinite(y)):
                raise ValueError(f"Invalid network predictions: {name}/{role}")
            path = folder / f"predictions_{name}_{role}.npz"
            np.savez_compressed(
                path, prediction=y, reference=labels, **{key: data[key] for key in PREDICTION_FIELDS},
            )
            hashes[path.name] = sha256(path)
            metrics = regression_metrics(y, labels)
            maes = metrics.pop("mae_per_variable")
            rows.append({
                "name": name, "model": kind or name, "seed": seed, "split": role, **metrics,
                **{f"mae_{key}": value for key, value in zip(("mb", "m1", "m2", "s"), maes)},
            })
    save_json(output / "predictions_frozen.json", {
        "training_status_sha256": sha256(output / "training_status.json"),
        "prediction_sha256": hashes, "case_key": list(CASE_FIELDS),
    })
    save_csv(output / "regression_metrics.csv", rows)
    return [name for name, _, _ in variants], hashes, rows


def run_pilot(source, output, config, device="cpu", samples=100000, bank_seed=1, alpha=.05):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Dataset and output must be separate directories")
    if type(samples) is not int or samples < 2 or type(bank_seed) is not int or bank_seed < 0 or not 0 < alpha < 1:
        raise ValueError("Need at least two audit samples, a nonnegative bank seed and 0 < alpha < 1")
    config = validate_training_config(deepcopy(config))
    source_status, schema = validate_source(source)
    source_plan = read_json(source / "plan.json")
    graphs = source / "graphs"
    normalizer = read_json(graphs / "normalization.json")
    bounds = source_plan["base_config"]["bounds"]
    splits = {role: load_split(graphs, role) for role in ("train", "validation")}
    plan = {
        "step": "6b", "scope": "development_pilot", "source_completion_sha256": sha256(source / "completion.json"),
        "source_plan_sha256": sha256(source / "plan.json"), "dataset_sha256": file_fingerprints(graphs),
        "case_key": list(CASE_FIELDS), "config": config, "device": device,
        "label_policy": "all validation-selected labels; heldout flags recorded only",
        "selection_metric": "validation_allocation_mse", "test_used_for_selection": False,
        "audit": {"samples": samples, "bank_seed": bank_seed, "alpha": alpha, "stream": AUDIT_STREAM},
        "code_sha256": {name: sha256(ROOT / name) for name in CODE_FILES},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "torch": str(torch.__version__)},
    }
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "training_plan.json", plan)

    runs = []
    for kind in config["models"]:
        for seed in config["seeds"]:
            name = f"{kind}_seed_{seed}"
            run = train_one(kind, seed, config, bounds, splits["train"], splits["validation"],
                            normalizer, schema, plan["dataset_sha256"], output / "training" / name, device)
            runs.append({"name": name, **run})
    chosen = {
        kind: min((r for r in runs if r["model"] == kind), key=lambda r: (r["validation"]["mse"], r["seed"]))["name"]
        for kind in config["models"]
    }
    status = {
        "step": "6b", "training_complete": True, "runs": runs, "selected_by_validation": chosen,
        "checkpoint_sha256": {run["name"]: run["checkpoint_sha256"] for run in runs},
        "test_used_for_training_or_selection": False,
    }
    save_json(output / "training_status.json", status)
    print(f"Frozen {len(runs)} checkpoints; selected by validation: {chosen}", flush=True)

    splits["test"] = load_split(graphs, "test")
    names, hashes, regression = save_predictions(output, splits, status, plan)
    physics_rows, audit = audit_predictions(
        source_plan, splits, names, output / "predictions", hashes, samples, bank_seed, alpha,
    )
    save_csv(output / "physics_metrics.csv", physics_rows)
    physics = []
    for role in ("validation", "test"):
        for name in (*names, "pso_reference"):
            group = [row for row in physics_rows if row["model"] == name and row["split"] == role]
            passed = [row for row in group if row["bound_checks_pass"]]
            physics.append({
                "model": name, "split": role, "cases": len(group), "bound_pass_count": len(passed),
                "bob_rate_mean": float(np.mean([row["bob_rate"] for row in group])),
                "bob_rate_mean_passing": float(np.mean([row["bob_rate"] for row in passed])) if passed else None,
                "min_dep_lower_margin": min(row["dep_lower_margin"] for row in group),
            })
    save_csv(output / "physics_summary.csv", physics)
    aggregate = []
    for role in SPLITS:
        for kind in ("train_mean", *config["models"]):
            group = [row for row in regression if row["model"] == kind and row["split"] == role]
            errors = [row["mse"] for row in group]
            aggregate.append({
                "model": kind, "split": role, "runs": len(group),
                "mse_mean": float(np.mean(errors)),
                "mse_sd_across_seeds": float(np.std(errors, ddof=1)) if len(group) > 1 else 0.,
                "mae_mean": float(np.mean([row["mae"] for row in group])),
            })
    save_csv(output / "regression_by_model.csv", aggregate)

    if (file_fingerprints(graphs) != plan["dataset_sha256"]
            or sha256(source / "plan.json") != plan["source_plan_sha256"]
            or sha256(source / "completion.json") != plan["source_completion_sha256"]):
        raise ValueError("The source dataset changed during step 6b")
    if any(sha256(output / "predictions" / name) != digest for name, digest in hashes.items()):
        raise ValueError("Predictions changed during the audit")
    result = {
        "step": "6b", "complete": True, "scope": "development_pilot",
        "splits": source_status["splits"], "layout_ids": {
            role: sorted(set(data["layout_id"].tolist())) for role, data in splits.items()
        },
        "checkpoints": len(runs), "selected_by_validation": chosen,
        "training_seconds_sum": sum(r["training_seconds_including_epoch_metrics"] for r in runs),
        "regression": regression, "regression_by_model": aggregate, "physics": physics, "audit": audit,
        "test_used_for_training_or_selection": False, "heldout_used_for_label_filter": False,
    }
    save_json(output / "summary.json", result)
    save_json(output / "completion.json", {"sha256": {
        path.relative_to(output).as_posix(): sha256(path) for path in sorted(output.rglob("*")) if path.is_file()
    }})
    return result


def check_output(output):
    output = Path(output).resolve()
    files = read_json(output / "completion.json")["sha256"]
    if "summary.json" not in files or "predictions_frozen.json" not in files:
        raise ValueError("Incomplete step 6b output")
    for name, digest in files.items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Step 6b result changed: {name}")
    summary = read_json(output / "summary.json")
    if summary["step"] != "6b" or not summary["complete"]:
        raise ValueError("Incomplete step 6b output")
    return summary
