"""Fixed-budget 7f supervised fits, CPU optimizer/RNG resume, frozen snapshots."""
from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import torch

from .expanded_data import write_table
from .learning_data import read_json, sha256
from .learning_train import configure_runtime
from .multik_dataset import write_json
from .multik_graph import pack_graphs
from .multik_models import balanced_allocation_loss, tensor_batch
from .continuous_learning_models import build_model, checkpoint, error_metrics, load_model, save_torch


def packed(rows, normalizer):
    return tensor_batch(pack_graphs([r["graph"] for r in rows], normalizer, [r["allocation"] for r in rows]))


def subset(tensors, indices):
    part = {key: value[indices] for key, value in tensors.items()}
    return part, part.pop("target")


def evaluate(model, tensors, config):
    model.eval()
    n = len(tensors["node"])
    total = {}
    with torch.inference_mode():
        for start in range(0, n, config["evaluation_batch_size"]):
            end = min(n, start + config["evaluation_batch_size"])
            batch, target = subset(tensors, slice(start, end))
            values = error_metrics(model(**batch), target, batch["node_mask"], config["loss_weights"])
            for key, value in values.items():
                total[key] = total.get(key, 0.) + value * (end-start) / n
    if not total or not all(np.isfinite(v) for v in total.values()):
        raise ArithmeticError("Nonfinite or empty allocation validation")
    return total


def history_row(epoch, train, validation):
    return {"epoch": epoch, **{"train_"+k:v for k,v in train.items()},
            **{"validation_"+k:v for k,v in validation.items()}}


def fit_folder(output, kind, seed):
    return Path(output) / "fits" / f"{kind}_s{seed}"


def read_fit(output, kind, seed, plan):
    folder = fit_folder(output, kind, seed)
    manifest = read_json(folder / "fit_manifest.json")
    if manifest["plan_sha256"] != plan["plan_sha256"] or manifest["kind"] != kind or manifest["seed"] != seed:
        raise ValueError("Fit manifest belongs to another 7f plan")
    expected={f'{kind}_s{seed}_e{e:04d}' for e in plan['config']['snapshot_epochs']}|{f'{kind}_s{seed}_best'}
    if set(manifest['candidates'])!=expected or manifest['epochs_completed']!=plan['config']['epochs']:
        raise ValueError('Fit checkpoint/epoch budget differs')
    for name, digest in manifest["sha256"].items():
        path = (folder/name).resolve()
        if not path.is_relative_to(folder.resolve()) or sha256(path) != digest:
            raise ValueError("A completed 7f fit changed")
    for ref in manifest["candidates"].values():
        path = (Path(output)/ref["path"]).resolve()
        if not path.is_relative_to(Path(output).resolve()) or sha256(path) != ref["sha256"]:
            raise ValueError("A frozen 7f checkpoint changed")
        network, saved = load_model(path)
        if (saved["plan_sha256"] != plan["plan_sha256"] or saved["model_kind"] != kind or saved["seed"] != seed
                or saved["normalization"] != plan["normalization"] or saved["epoch"] != ref["epoch"]
                or saved['training_config'] != plan['config']
                or saved['source_7e_completion_sha256'] != plan['dataset_sha256']['completion.json']
                or saved['validation_errors'] != ref['validation_errors']):
            raise ValueError("7f checkpoint metadata differs from its fit")
        parameters=sum(p.numel() for p in network.parameters())
        if parameters!=ref['parameters'] or parameters!=manifest['parameters']:
            raise ValueError('Model parameter count differs from frozen fit')
    return manifest


def fit(output, kind, seed, plan, bounds, train_rows, validation_rows, stop_after_epoch=None):
    folder = fit_folder(output, kind, seed)
    if (folder/"fit_manifest.json").exists():
        return read_fit(output, kind, seed, plan)
    folder.mkdir(parents=True, exist_ok=True)
    c = plan["config"]
    configure_runtime(seed, 1, "cpu")
    model = build_model(kind, bounds, c)
    optimizer = torch.optim.Adam(model.parameters(), lr=c["learning_rate"], weight_decay=c["weight_decay"])
    random = np.random.default_rng(seed)
    train = packed(train_rows, plan["normalization"])
    validation = packed(validation_rows, plan["normalization"])
    pointer = folder/"resume.json"
    seconds_before = 0.
    if pointer.exists():
        record = read_json(pointer)
        path = (folder/record["path"]).resolve()
        if not path.is_relative_to(folder.resolve()) or sha256(path) != record["sha256"]:
            raise ValueError("7f optimizer resume file missing/changed")
        resume = torch.load(path, map_location="cpu", weights_only=True)
        if (resume["format"] != "continuous_learning_resume_7f_v1" or resume["plan_sha256"] != plan["plan_sha256"]
                or resume["kind"] != kind or resume["seed"] != seed):
            raise ValueError("7f optimizer resume belongs to another plan")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        random.bit_generator.state = resume["numpy_rng"]
        torch.set_rng_state(resume["torch_rng"])
        history, best_state, best_epoch, best_errors = resume["history"], resume["best_state"], resume["best_epoch"], resume["best_errors"]
        start_epoch = resume["epoch"]
        seconds_before = resume["training_seconds"]
        if [r["epoch"] for r in history] != list(range(start_epoch+1)):
            raise ValueError("Resume history has missing or duplicate epochs")
        print(f"7f resume {kind} seed={seed} from epoch={start_epoch}", flush=True)
    else:
        train_error, val_error = evaluate(model,train,c), evaluate(model,validation,c)
        history = [history_row(0, train_error, val_error)]
        best_state, best_epoch, best_errors = deepcopy(model.state_dict()), 0, val_error
        start_epoch = 0
    started = time.perf_counter()

    def save_progress(epoch):
        path = folder/f"resume_e{epoch:04d}.pt"
        save_torch(path, {"format":"continuous_learning_resume_7f_v1", "plan_sha256":plan["plan_sha256"],
                   "kind":kind,"seed":seed,"epoch":epoch,"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                   "numpy_rng":random.bit_generator.state,"torch_rng":torch.get_rng_state(),"history":history,
                   "best_state":best_state,"best_epoch":best_epoch,"best_errors":best_errors,
                   "training_seconds":seconds_before+time.perf_counter()-started})
        # Commit the pointer after its bytes. Keep the old file until this succeeds.
        write_json(pointer,{"path":path.name,"sha256":sha256(path),"epoch":epoch})
        for old in folder.glob("resume_e*.pt"):
            if old != path:
                old.unlink()

    for epoch in range(start_epoch+1,c["epochs"]+1):
        model.train()
        indices = random.permutation(len(train_rows))
        for start in range(0,len(indices),c["batch_size"]):
            batch,target = subset(train,indices[start:start+c["batch_size"]])
            optimizer.zero_grad(set_to_none=True)
            weights = c["loss_weights"]
            loss = balanced_allocation_loss(model(**batch),target,batch["node_mask"],weights["mb"],weights["public_total"],weights["s"])
            if not torch.isfinite(loss):
                raise ArithmeticError("Nonfinite mixed-K supervised loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),c["gradient_clip"],error_if_nonfinite=True)
            optimizer.step()
        train_error, val_error = evaluate(model,train,c), evaluate(model,validation,c)
        history.append(history_row(epoch,train_error,val_error))
        if val_error["weighted_loss"] < best_errors["weighted_loss"]:
            best_state,best_epoch,best_errors = deepcopy(model.state_dict()),epoch,val_error
        if epoch in c["snapshot_epochs"]:
            save_torch(folder/"snapshots"/f"epoch_{epoch:04d}.pt",checkpoint(model,model.state_dict(),kind,seed,epoch,plan,bounds,val_error))
        if epoch%c["resume_every"]==0 or epoch in c["snapshot_epochs"] or epoch==c["epochs"] or epoch==stop_after_epoch:
            save_progress(epoch)
        if epoch==1 or epoch%20==0 or epoch==c["epochs"]:
            print(f"7f {kind} seed={seed} epoch={epoch}: train={train_error['weighted_loss']:.6g} val={val_error['weighted_loss']:.6g}",flush=True)
        if epoch==stop_after_epoch:
            raise InterruptedError("Simulated interruption after a committed optimizer checkpoint")
    save_torch(folder/"best_weighted.pt",checkpoint(model,best_state,kind,seed,best_epoch,plan,bounds,best_errors))
    write_table(folder/"history.csv",history)
    refs = {}
    paths = [(f"{kind}_s{seed}_e{e:04d}",folder/"snapshots"/f"epoch_{e:04d}.pt") for e in c["snapshot_epochs"]]
    paths.append((f"{kind}_s{seed}_best",folder/"best_weighted.pt"))
    for key,path in paths:
        saved = torch.load(path,map_location="cpu",weights_only=True)
        refs[key] = {"kind":kind,"seed":seed,"epoch":saved["epoch"],"path":path.relative_to(output).as_posix(),
                     "sha256":sha256(path),"validation_errors":saved["validation_errors"],
                     "parameters":sum(p.numel() for p in model.parameters())}
    write_json(folder/"fit_manifest.json",{"kind":kind,"seed":seed,"epochs_completed":c["epochs"],
               "plan_sha256":plan["plan_sha256"],"best_weighted_epoch":best_epoch,"parameters":sum(p.numel() for p in model.parameters()),
               "training_seconds":seconds_before+time.perf_counter()-started,"candidates":refs,
               "sha256":{p.relative_to(folder).as_posix():sha256(p) for _,p in paths}|{"history.csv":sha256(folder/"history.csv")},
               "test_used_for_training_or_selection":False,"physical_gradient":False})
    # Completed fits need the frozen weights/history, not optimizer state.
    pointer.unlink(missing_ok=True)
    for old in folder.glob("resume_e*.pt"):
        old.unlink()
    return read_fit(output,kind,seed,plan)
