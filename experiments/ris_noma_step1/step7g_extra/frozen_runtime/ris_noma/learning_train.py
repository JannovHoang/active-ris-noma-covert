"""Train with train graphs, select with validation MSE, freeze before test.

No physics-based gradient, audit filtering, test-based early stopping, or PSO
optimization occurs in this module. All declared seeds are kept and reported.
"""
from copy import deepcopy
from pathlib import Path
import csv
import os
import platform
import random
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .learning_data import (INPUT_KEYS, TARGET_KEY, inspect_export, load_split,
                            file_fingerprints, read_json, write_json, sha256)
from .learning_models import make_model

TRAINING_CODE = ["run_step5b.py", "ris_noma/learning_data.py", "ris_noma/learning_models.py",
                 "ris_noma/learning_train.py", "ris_noma/learning_physics.py"]


def validate_training_config(c):
    expected = {"models", "seeds", "epochs", "batch_size", "learning_rate", "weight_decay", "patience",
                "min_delta", "gnn_hidden", "gnn_layers", "mlp_hidden", "mlp_layers", "gradient_clip", "threads", "notes"}
    if set(c) != expected:
        raise ValueError(f"Training config fields differ: {set(c) ^ expected}")
    if not c["models"] or len(set(c["models"])) != len(c["models"]) or any(m not in ["gnn", "mlp"] for m in c["models"]):
        raise ValueError("Use distinct gnn/mlp models")
    if not c["seeds"] or len(set(c["seeds"])) != len(c["seeds"]) or any(type(s) is not int or s < 0 for s in c["seeds"]):
        raise ValueError("Seeds must be distinct nonnegative integers")
    for key in ["epochs", "batch_size", "patience", "gnn_hidden", "gnn_layers", "mlp_hidden", "mlp_layers", "threads"]:
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ["learning_rate", "weight_decay", "min_delta", "gradient_clip"]:
        if not np.isfinite(c[key]) or c[key] < 0 or (key in ["learning_rate", "gradient_clip"] and c[key] == 0):
            raise ValueError(f"Invalid {key}")
    return c


def configure_runtime(seed, threads=1, device="cpu"):
    if device not in ["cpu", "cuda", "auto"]:
        raise ValueError("device must be cpu, cuda, or auto")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    chosen = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    if chosen == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; choose --device cpu")
    return torch.device(chosen)


def tensors(data, device):
    return {k: torch.as_tensor(data[k], dtype=torch.long if k == "edge_index" else torch.float32, device=device)
            for k in [*INPUT_KEYS, TARGET_KEY]}


def predict(model, data, device="cpu", batch_size=256):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(data["node"]), batch_size):
            part = {k: data[k][start:start+batch_size] for k in [*INPUT_KEYS, TARGET_KEY]}
            t = tensors(part, device)
            predictions.append(model(**{k: t[k] for k in INPUT_KEYS}).cpu().numpy())
    return np.concatenate(predictions).astype(float) if predictions else np.empty((0, 4))


def regression_metrics(predictions, labels):
    if not len(labels):
        return {"n": 0, "mse": None, "mae": None, "mae_per_variable": None, "max_absolute_error": None}
    delta = np.asarray(predictions) - labels
    return {"n": len(labels), "mse": float(np.mean(delta ** 2)), "mae": float(np.mean(np.abs(delta))),
            "mae_per_variable": np.abs(delta).mean(axis=0).tolist(), "max_absolute_error": float(np.abs(delta).max())}


def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def plot_history(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for key, label in [("train_mse", "Train MSE"), ("validation_mse", "Validation MSE")]:
        ax.plot([r["epoch"] for r in rows], [r[key] for r in rows], label=label)
    ax.set(xlabel="Epoch", ylabel="Mean squared allocation error", yscale="log")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def train_one(kind, seed, c, bounds, train, validation, normalizer, schema, dataset_fingerprints, folder, device="cpu"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    device = configure_runtime(seed, c["threads"], device)
    model = make_model(kind, bounds, c).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=c["learning_rate"], weight_decay=c["weight_decay"])
    cpu_train = tensors(train, "cpu")
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(*[cpu_train[k] for k in [*INPUT_KEYS, TARGET_KEY]]),
                        batch_size=c["batch_size"], shuffle=True, generator=generator, num_workers=0)
    loss_function = nn.MSELoss()
    initial_train = regression_metrics(predict(model, train, device), train[TARGET_KEY])["mse"]
    initial_validation = regression_metrics(predict(model, validation, device), validation[TARGET_KEY])["mse"]
    best_loss, best_epoch, best_state = initial_validation, 0, deepcopy(model.state_dict())
    significant_best, stale = initial_validation, 0
    history, start = [], time.perf_counter()
    for epoch in range(1, c["epochs"] + 1):
        model.train()
        for batch in loader:
            inputs = {k: batch[j].to(device) for j, k in enumerate(INPUT_KEYS)}
            target = batch[-1].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(**inputs), target)
            if not torch.isfinite(loss):
                raise ArithmeticError("Nonfinite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), c["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
        train_loss = regression_metrics(predict(model, train, device), train[TARGET_KEY])["mse"]
        val_loss = regression_metrics(predict(model, validation, device), validation[TARGET_KEY])["mse"]
        history.append({"epoch": epoch, "train_mse": train_loss, "validation_mse": val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch, best_state = val_loss, epoch, deepcopy(model.state_dict())
        if val_loss < significant_best - c["min_delta"]:
            significant_best, stale = val_loss, 0
        else:
            stale += 1
        if epoch == 1 or epoch % 50 == 0:
            print(f"{kind} seed={seed} epoch={epoch}: train={train_loss:.6g} val={val_loss:.6g}", flush=True)
        if stale >= c["patience"]:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    model.load_state_dict(best_state)
    checkpoint = {"format": "ris_noma_step5b_v1", "model_kind": kind, "seed": seed, "config": c,
                  "bounds": bounds, "feature_schema": schema, "normalization": normalizer,
                  "dataset_sha256": dataset_fingerprints, "best_epoch": best_epoch,
                  "selection_metric": "validation_allocation_mse", "test_used_for_selection": False,
                  "state_dict": {k: v.detach().cpu() for k, v in best_state.items()}}
    torch.save(checkpoint, folder / "best_model.pt")
    save_csv(folder / "history.csv", history)
    plot_history(folder / "learning_curve.png", history)
    result = {"model": kind, "seed": seed, "parameters": sum(p.numel() for p in model.parameters()),
              "device": str(device), "epochs_completed": len(history), "best_epoch": best_epoch,
              "initial_train_mse": initial_train, "initial_validation_mse": initial_validation,
              "train": regression_metrics(predict(model, train, device), train[TARGET_KEY]),
              "validation": regression_metrics(predict(model, validation, device), validation[TARGET_KEY]),
              "training_seconds_including_epoch_metrics": elapsed, "test_used_for_selection": False,
              "checkpoint_sha256": sha256(folder / "best_model.pt")}
    write_json(folder / "training_summary.json", result)
    return result


def train_experiment(dataset, output, config, device="cpu"):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if dataset == output or dataset in output.parents or output in dataset.parents:
        raise ValueError("Dataset and training output must be separate, non-nested directories")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Training output is nonempty; use a new output directory")
    c = validate_training_config(deepcopy(config))
    report = inspect_export(dataset)
    if not report["ready_for_training_pilot"]:
        raise ValueError("Need nonempty train and validation splits")
    train, val = load_split(dataset, "train"), load_split(dataset, "validation")
    manifest = read_json(dataset / "dataset_manifest.json")
    schema, normalizer = read_json(dataset / "feature_schema.json"), read_json(dataset / "normalization.json")
    code_root = Path(__file__).resolve().parents[1]
    output.mkdir(parents=True, exist_ok=True)
    plan = {"step": "5b", "dataset_path_at_run": str(dataset), "dataset_sha256": report["input_files_sha256"],
            "config": c, "requested_device": device, "selection": "minimum validation MSE; early stopping on validation only",
            "audit_policy": "keep all validation-selected reference labels; never filter by heldout outcome",
            "test_used_for_training_or_checkpoint_selection": False, "final_gnn_test": False,
            "code_sha256": {n: sha256(code_root / n) for n in TRAINING_CODE},
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__)}}
    write_json(output / "training_plan.json", plan)
    write_json(output / "dataset_inspection.json", report)
    rows = []
    for kind in c["models"]:
        for seed in c["seeds"]:
            rows.append(train_one(kind, seed, c, manifest["bounds"], train, val, normalizer, schema,
                                  plan["dataset_sha256"], output / f"{kind}_seed_{seed}", device))
    checkpoint_hashes = {f"{r['model']}_seed_{r['seed']}/best_model.pt": r["checkpoint_sha256"] for r in rows}
    status = {"step": "5b", "training_complete": True, "trained_model": True, "runs": rows,
              "checkpoint_sha256": checkpoint_hashes, "test_evaluated": False,
              "all_declared_seeds_reported": True, "final_gnn_test": False,
              "input_files_unchanged": file_fingerprints(dataset) == plan["dataset_sha256"]}
    if not status["input_files_unchanged"]:
        raise ValueError("Source dataset changed during training")
    write_json(output / "training_status.json", status)
    return status


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "ris_noma_step5b_v1" or checkpoint["test_used_for_selection"]:
        raise ValueError("Unsupported checkpoint or test-based selection")
    model = make_model(checkpoint["model_kind"], checkpoint["bounds"], checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device); model.eval()
    return model, checkpoint


def evaluate_experiment(dataset, training, output):
    """Post-freeze pilot evaluation; no selection of a seed by test results."""
    dataset, training, output = Path(dataset), Path(training), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Evaluation output is nonempty; use a new output directory")
    plan, status = read_json(training / "training_plan.json"), read_json(training / "training_status.json")
    if not status["training_complete"] or file_fingerprints(dataset) != plan["dataset_sha256"]:
        raise ValueError("Need frozen training and exactly the same dataset")
    for name, digest in status["checkpoint_sha256"].items():
        if sha256(training / name) != digest:
            raise ValueError("Checkpoint changed after training freeze")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "evaluation_plan.json", {"dataset_sha256": plan["dataset_sha256"],
               "checkpoint_sha256": status["checkpoint_sha256"], "selection_on_test": False,
               "scope": "pilot test evaluated only after all declared runs froze; repeated use makes this development data"})
    splits = {role: load_split(dataset, role) for role in ["train", "validation", "test"]}
    configure_runtime(0, plan["config"]["threads"], "cpu")
    rows, prediction_sets = [], {}
    baseline = splits["train"][TARGET_KEY].mean(axis=0)
    for role, data in splits.items():
        y = np.broadcast_to(baseline, data[TARGET_KEY].shape).copy()
        rows.append({"model": "train_mean", "seed": None, "split": role, **regression_metrics(y, data[TARGET_KEY])})
        prediction_sets[("train_mean", role)] = y
    for run in status["runs"]:
        name = f"{run['model']}_seed_{run['seed']}"
        model, checkpoint = load_checkpoint(training / name / "best_model.pt")
        if checkpoint["dataset_sha256"] != plan["dataset_sha256"]:
            raise ValueError("Checkpoint trained on another dataset")
        for role, data in splits.items():
            y = predict(model, data)
            prediction_sets[(name, role)] = y
            rows.append({"model": run["model"], "seed": run["seed"], "split": role, **regression_metrics(y, data[TARGET_KEY])})
    for (name, role), y in prediction_sets.items():
        data = splits[role]
        np.savez_compressed(output / f"predictions_{name}_{role}.npz", prediction=y,
                            reference=data[TARGET_KEY], layout_id=data["layout_id"], xi=data["xi"],
                            sic_order_original_ids=data["sic_order_original_ids"])
    save_csv(output / "regression_metrics.csv", [{**r, "mae_per_variable": str(r["mae_per_variable"])} for r in rows])
    aggregates = []
    for kind in plan["config"]["models"]:
        for role in splits:
            subset = [r for r in rows if r["model"] == kind and r["split"] == role]
            aggregates.append({"model": kind, "split": role, "runs": len(subset),
                               "mse_mean": float(np.mean([r["mse"] for r in subset])),
                               "mse_sd_across_training_seeds": float(np.std([r["mse"] for r in subset], ddof=1)) if len(subset) > 1 else 0.})
    summary = {"pilot_evaluation_complete": True, "regression": rows, "aggregate": aggregates,
               "test_used_for_training_or_selection": False, "final_gnn_test": False,
               "scope": "allocation errors only; physical feasibility requires independent forward evaluation"}
    write_json(output / "evaluation_summary.json", summary)
    return summary
