"""Allocation-only weighted training; fixed checkpoint pool, no physics gradients."""
from copy import deepcopy
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .dep_label_learning import read_model, write_immutable
from .expanded_data import save_json
from .learning_data import INPUT_KEYS, TARGET_KEY, read_json, sha256
from .learning_models import make_model
from .learning_train import configure_runtime, predict, regression_metrics, save_csv, tensors


def weighted_mse(prediction, target, weights):
    """Mean over cases of sum_j(w_j*error_j**2)/sum_j(w_j)."""
    weights = torch.as_tensor(weights, dtype=prediction.dtype, device=prediction.device)
    if (prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[-1] != 4
            or weights.shape != (4,) or not torch.isfinite(weights).all() or not (weights > 0).all()):
        raise ValueError("Need equal N x 4 allocations and four finite positive weights")
    if bool(torch.all(weights == 1)):
        return nn.functional.mse_loss(prediction, target)
    return ((prediction - target).square() * weights).sum(-1).mean() / weights.sum()


def state_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def make_checkpoint(model, state, seed, epoch, plan, bounds, loss, validation_mse, is_mse_best):
    return {"format": "ris_noma_step5b_v1", "model_kind": "gnn", "seed": seed,
            "config": plan["training_config"], "bounds": bounds,
            "feature_schema": plan["feature_schema"], "normalization": plan["normalization"],
            "dataset_sha256": plan["sources_sha256"]["previous_inputs"],
            "best_epoch": epoch, "selection_metric": "validation_allocation_mse" if is_mse_best else "predeclared_epoch",
            "test_used_for_selection": False, "step6f_loss": loss, "loss_weights": plan["loss_weights"][loss],
            "validation_mse": validation_mse,
            "state_dict": {k: v.detach().cpu().clone() for k, v in state.items()}}


def train_fit(folder, relative, plan, bounds, data, loss, seed):
    folder.mkdir()
    c, weights = plan["training_config"], plan["loss_weights"][loss]
    device = configure_runtime(seed, c["threads"], "cpu")
    model = make_model("gnn", bounds, c).to(device)
    initial_hash = state_fingerprint(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=c["learning_rate"], weight_decay=c["weight_decay"])
    train, validation = data["train"], data["validation"]
    tensor_data = tensors(train, "cpu")
    loader = DataLoader(TensorDataset(*[tensor_data[k] for k in (*INPUT_KEYS, TARGET_KEY)]),
                        batch_size=c["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    initial_y = predict(model, validation)
    initial_mse = regression_metrics(initial_y, validation[TARGET_KEY])["mse"]
    best_mse, best_epoch, best_state = initial_mse, 0, deepcopy(model.state_dict())
    history, refs, started = [], {}, time.perf_counter()
    for epoch in range(1, c["epochs"] + 1):
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(**{key: batch[j] for j, key in enumerate(INPUT_KEYS)})
            value = weighted_mse(prediction, batch[-1], weights)
            if not torch.isfinite(value):
                raise ArithmeticError("Nonfinite step 6f training loss")
            value.backward()
            nn.utils.clip_grad_norm_(model.parameters(), c["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
        train_y, val_y = predict(model, train), predict(model, validation)
        train_mse = regression_metrics(train_y, train[TARGET_KEY])["mse"]
        val_mse = regression_metrics(val_y, validation[TARGET_KEY])["mse"]
        def loss_value(y, labels):
            return float(np.mean(np.sum((y - labels) ** 2 * weights, axis=-1)) / sum(weights))
        history.append({"epoch": epoch, "train_mse": train_mse, "validation_mse": val_mse,
                        "train_weighted_loss": loss_value(train_y, train[TARGET_KEY]),
                        "validation_weighted_loss": loss_value(val_y, validation[TARGET_KEY]),
                        "train_mb_bias": float(np.mean(train_y[:, 0] - train[TARGET_KEY][:, 0])),
                        "validation_mb_bias": float(np.mean(val_y[:, 0] - validation[TARGET_KEY][:, 0]))})
        if val_mse < best_mse:
            best_mse, best_epoch, best_state = val_mse, epoch, deepcopy(model.state_dict())
        if epoch in plan["snapshot_epochs"]:
            name = f"epoch_{epoch:04d}.pt"
            torch.save(make_checkpoint(model, model.state_dict(), seed, epoch, plan, bounds, loss, val_mse, False), folder / name)
            refs[f"{loss}_s{seed}_e{epoch:04d}"] = {
                "path": f"{relative}/{name}", "sha256": sha256(folder / name), "loss": loss, "seed": seed,
                "epoch": epoch, "validation_mse": val_mse, "is_mse_best": False}
        if epoch == 1 or epoch % 50 == 0:
            print(f"6f {loss} seed={seed} epoch={epoch}: train MSE={train_mse:.6g}; val MSE={val_mse:.6g}", flush=True)
    torch.save(make_checkpoint(model, best_state, seed, best_epoch, plan, bounds, loss, best_mse, True), folder / "best_model.pt")
    refs[f"{loss}_s{seed}_mse"] = {
        "path": f"{relative}/best_model.pt", "sha256": sha256(folder / "best_model.pt"), "loss": loss,
        "seed": seed, "epoch": best_epoch, "validation_mse": best_mse, "is_mse_best": True}
    save_csv(folder / "history.csv", history)
    manifest = {"step": "6f", "loss": loss, "weights": weights, "seed": seed,
                "initial_state_sha256": initial_hash, "initial_validation_mse": initial_mse,
                "epochs_completed": c["epochs"], "early_stopping": False, "best_mse_epoch": best_epoch,
                "parameters": sum(p.numel() for p in model.parameters()),
                "training_seconds": time.perf_counter() - started,
                "history_sha256": sha256(folder / "history.csv"), "candidates": refs,
                "plan_sha256": plan["fit_plan_sha256"], "test_used_for_selection": False}
    save_json(folder / "fit_manifest.json", manifest)
    return manifest


def fit_models(output, plan, bounds, data):
    write_immutable(output / "learning_plan.json", {
        "step": "6f", "plan_sha256": sha256(output / "plan.json"),
        "config": plan["training_config"], "loss_weights": plan["loss_weights"],
        "snapshot_epochs": plan["snapshot_epochs"], "label_slack": plan["label_slack"],
        "fixed_epochs_for_both_losses": True, "early_stopping": False,
        "normalization_fitted_on": "6e train only; reused unchanged",
        "physical_gradient": False, "test_used_for_training_or_selection": False})
    fit_plan = dict(plan, fit_plan_sha256=sha256(output / "learning_plan.json"))
    folder = output / "fits"
    folder.mkdir(exist_ok=True)
    refs, manifests = {}, {}
    for loss in plan["loss_weights"]:
        for seed in plan["training_config"]["seeds"]:
            name = f"{loss}_s{seed}"
            target = folder / name
            if not target.exists():
                with TemporaryDirectory(prefix=".fit_", dir=output) as temporary:
                    stage = Path(temporary) / "model"
                    train_fit(stage, f"fits/{name}", fit_plan, bounds, data, loss, seed)
                    stage.replace(target)
            record = read_json(target / "fit_manifest.json")
            if (record["plan_sha256"] != fit_plan["fit_plan_sha256"] or record["loss"] != loss
                    or record["seed"] != seed or record["weights"] != plan["loss_weights"][loss]
                    or record["epochs_completed"] != plan["epochs"]
                    or sha256(target / "history.csv") != record["history_sha256"]):
                raise ValueError(f"Fit manifest differs from the current plan: {name}")
            for ref in record["candidates"].values():
                _, checkpoint = read_model(ref, output)
                if (checkpoint["dataset_sha256"] != plan["sources_sha256"]["previous_inputs"]
                        or checkpoint["normalization"] != plan["normalization"] or checkpoint["seed"] != seed
                        or checkpoint["step6f_loss"] != loss or checkpoint["config"] != plan["training_config"]):
                    raise ValueError(f"Checkpoint differs from the training plan: {name}")
            refs.update(record["candidates"])
            manifests[name] = sha256(target / "fit_manifest.json")
            print(f"Fit {len(manifests)}/{len(plan['loss_weights']) * len(plan['training_config']['seeds'])} committed: {name}", flush=True)
    for seed in plan["training_config"]["seeds"]:
        paired = [read_json(folder / f"{loss}_s{seed}/fit_manifest.json")["initial_state_sha256"]
                  for loss in plan["loss_weights"]]
        if len(set(paired)) != 1:
            raise ValueError("Paired losses must start from identical initial model weights")
    if len(refs) != plan["max_validation_candidates"]:
        raise ValueError("Checkpoint pool size differs from the predeclared validation budget")
    write_immutable(output / "fits_frozen.json", {"candidates": refs, "manifests": manifests,
                    "learning_plan_sha256": fit_plan["fit_plan_sha256"]})
    return refs
