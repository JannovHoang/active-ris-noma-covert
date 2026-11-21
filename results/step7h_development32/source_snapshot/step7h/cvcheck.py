"""Exploratory leave-one-low-CV-layout-out; main deployment checkpoints never change."""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from shutil import copy2
import time
import numpy as np

from .common import read, lines, write, digest, object_hash, commit, committed, freeze, verify, table
from .physics import context, corrected
from .analysis import metrics
from .benchmark import execute_jobs
from ris_noma.curve_fit import fit
from ris_noma.curve_models import load_model
from ris_noma.multik_graph import fit_normalizer, statistical_graph
from ris_noma.multik_models import predict
from ris_noma.channels import public_order
from ris_noma.learning_train import configure_runtime
from ris_noma.stable_assessment import raw_assessment
from ris_noma.multik_data_core import at_target


def prepare(parent, main):
    parent = Path(parent)
    output = parent / "cv"
    output.mkdir(exist_ok=True)
    source = read(parent / "inputs/stage_plan.json")
    jobs = []
    original = read(parent / "inputs/corpus_plan.json")["jobs"]
    for old in original:
        if old["split"] != "train" or old["layout_id"] not in main["cv_layout_ids"]: continue
        job = deepcopy(old)
        job.update(id=f"cv_l{old['layout_id']:06d}_k{old['n_public']}", split="diagnostic", regime="low_cv_leave_layout_out")
        if main["settings"]["scope"] == "smoke": job["cases"] = job["cases"][:main["settings"]["smoke_cases_per_job"]]
        jobs.append(job)
    plan = {key: main[key] for key in ("format", "settings", "workers", "pso_settings", "code_sha256", "source_hashes",
        "certificate_pool_size", "certificate_alpha_per_group", "cv_layout_ids")}
    plan.update(stage="cv", jobs=jobs, parent_plan_sha256=digest(parent / "plan.json"),
        fold_policy="Hold out one entire low-CV training layout at a time; train on other 31 layouts; one fixed seed17; fixed final epoch200; no held-out selection",
        scope_note="Exploratory after inspecting low-CV layouts; not final test. These retrained diagnostic networks never replace main six checkpoints")
    write(output / "plan.json", plan, immutable=True)
    for name in ("stage_plan.json", "physics.json", "dataset.json", "corpus_plan.json", "train.jsonl", "validation.jsonl"):
        target = output / "inputs" / name
        target.parent.mkdir(exist_ok=True)
        if target.exists() and digest(target) != digest(parent / "inputs" / name): raise ValueError("CV source snapshot changed")
        if not target.exists(): copy2(parent / "inputs" / name, target)
    return output, plan


def fit_folds(output, plan):
    output = Path(output)
    source = read(output / "inputs/stage_plan.json")
    train, validation = (lines(output / f"inputs/{role}.jsonl") for role in ("train", "validation"))
    bounds = read(output / "inputs/physics.json")["base_config"]["bounds"]
    refs = {}
    for lid in plan["cv_layout_ids"]:
        training = [r for r in train if r["layout_id"] != lid and r["has_label"]]
        ids = sorted({r["layout_id"] for r in training})
        if len(ids) != 31 or lid in ids: raise ValueError("Leave-layout-out fold leaked held-out geometry")
        config = dict(source["config"]["training"], **source["config"]["architectures"]["small"],
            validation_physics_samples=plan["settings"]["audit_samples"], test_physics_samples=plan["settings"]["audit_samples"], alpha=plan["settings"]["alpha"])
        config.update(epochs=plan["settings"]["cv_epochs"], snapshot_epochs=[plan["settings"]["cv_epochs"]])
        config["resume_every"] = min(config["resume_every"], config["epochs"])
        fold = {"config": config, "normalization": fit_normalizer([r["graph"] for r in training]),
                "architecture_name": "small", "training_layout_ids": ids, "training_public_counts": [2, 3, 4, 5],
                "stage_completion_sha256": source["stage_completion_sha256"],
                "corpus_plan_sha256": source["corpus_plan_sha256"], "held_out_layout_id": lid}
        fold["plan_sha256"] = object_hash(fold)
        folder = output / "folds" / f"layout_{lid}"
        write(folder / "plan.json", fold, immutable=True)
        for kind in ("gnn", "mlp"):
            manifest = fit(folder, kind, plan["settings"]["cv_seed"], fold, bounds, training,
                           [r for r in validation if r["has_label"]])
            # Deliberately fixed final epoch: never select by held-out physical performance.
            key = f"{kind}_s{plan['settings']['cv_seed']}_e{config['epochs']:04d}"
            ref = manifest["candidates"][key]
            refs[f"{lid}::{kind}"] = dict(ref, path=(folder / ref["path"]).relative_to(output).as_posix(),
                normalization=fold["normalization"], training_layout_ids=ids, held_out_layout_id=lid,
                checkpoint_policy="fixed_final_epoch", fold_plan_sha256=digest(folder / "plan.json"))
    write(output / "fold_models.json", refs, immutable=True)
    return refs


def predict_folds(parent, output, plan, refs):
    parent, output = Path(parent), Path(output)
    deploy = read(parent / "models.json")
    source_rows = {r["case_id"]: r for r in lines(output / "inputs/train.jsonl")}
    paths = []
    for job in plan["jobs"]:
        path = output / f"predictions/{job['id']}.json"
        old = committed(path)
        if old is not None:
            if old["plan_sha256"] != digest(output / "plan.json"): raise ValueError("CV prediction plan changed")
            paths.extend((path, Path(str(path)+".receipt.json")))
            continue
        c, layout, bank, meta = context(output, plan, job, "cv_input")
        order = public_order(bank)
        networks = {kind: load_model(output / refs[f"{job['layout_id']}::{kind}"]["path"])[0] for kind in ("gnn", "mlp")}
        records = []
        for case in job["cases"]:
            started = time.perf_counter()
            graph = statistical_graph(dict(c, total_dbm=case["power_dbm"]), layout, bank, order, case["xi"])
            graph_seconds = time.perf_counter()-started
            for kind, model in networks.items():
                ref = refs[f"{job['layout_id']}::{kind}"]
                started = time.perf_counter()
                raw = predict(model, [graph], ref["normalization"])[0]
                forward_seconds = time.perf_counter()-started
                delta = deploy[f"{kind}_s{plan['settings']['cv_seed']}"]["delta"]
                source_id = f"train_l{job['layout_id']:06d}_k{job['n_public']}_c{case['case_index']:03d}"
                original = source_rows[source_id]
                for shift in (0., delta):
                    started = time.perf_counter()
                    y = corrected(raw, shift, c["bounds"])
                    online = meta["bank_generation_seconds"]/len(job["cases"])+graph_seconds+forward_seconds+time.perf_counter()-started
                    records.append({"case_id": f"{job['id']}_c{case['case_index']:03d}", "layout_id": job["layout_id"],
                        "n_public": job["n_public"], "regime": "low_cv_leave_layout_out", "replicate": plan["settings"]["cv_seed"],
                        "method": kind+"_direct", "budget": 0, "power_dbm": case["power_dbm"], "xi": case["xi"],
                        "allocation": y.tolist(), "delta": shift, "order": order.tolist(), "returned": True,
                        "objective_evaluations": 0, "pso_seconds": 0., "online_seconds": online,
                        "feature_cv_willie": graph["global_raw"][18], "has_label": original["has_label"],
                        "mb_error": float(y[0]-original["allocation"][0]) if original["has_label"] else None,
                        "held_out_layout_id": job["layout_id"], "training_layout_ids": ref["training_layout_ids"],
                        "checkpoint_sha256": ref["sha256"], "epoch": ref["epoch"]})
        commit(path, {"job": job, "rows": records, "metadata": meta,
                      "plan_sha256": digest(output / "plan.json"), "fold_models_sha256": digest(output / "fold_models.json")})
        paths.extend((path, Path(str(path)+".receipt.json")))
    freeze(output, "allocations_frozen.json", paths, plan_sha256=digest(output / "plan.json"))


def audit_task(args):
    output, job = args
    output = Path(output)
    configure_runtime(0, 1, "cpu")
    plan = read(output / "plan.json")
    path = output / f"audit/{job['id']}.json"
    if committed(path) is not None: return job["id"]
    prediction = committed(output / f"predictions/{job['id']}.json")
    c, layout, bank, meta = context(output, plan, job, "cv_audit")
    cache, rows = {}, []
    started = time.perf_counter()
    for row in prediction["rows"]:
        key = row["power_dbm"], tuple(row["allocation"])
        if key not in cache:
            cache[key] = raw_assessment(dict(c, total_dbm=row["power_dbm"]), layout, bank,
                np.asarray(row["order"], int), row["allocation"], plan["certificate_pool_size"], plan["certificate_alpha_per_group"])
        raw = cache[key]
        rows.append({**row, **raw, **at_target(raw, row["xi"]), "dep_lower_margin": raw["dep_lower"]-row["xi"],
                     "invalid_layout_assumption": not raw["point_checks_without_dep"]["layout_bob_mean_gain_rule"]})
    commit(path, {"job": job, "rows": rows, "metadata": meta, "scoring_seconds": time.perf_counter()-started,
                  "unique_allocation_evaluations": len(cache), "plan_sha256": digest(output / "plan.json"),
                  "allocations_sha256": digest(output / "allocations_frozen.json")})
    return job["id"]


def run(parent, main):
    output, plan = prepare(parent, main)
    if (output / "completion.json").exists():
        verify(output, read(output / "completion.json"))
        return read(output / "summary.json")
    refs = fit_folds(output, plan)
    predict_folds(parent, output, plan, refs)
    verify(output, read(output / "allocations_frozen.json"))
    pending = [j for j in plan["jobs"] if committed(output / f"audit/{j['id']}.json") is None]
    execute_jobs(audit_task, [(str(output), j) for j in pending], plan["workers"], "leave-low-CV-layout-out audit")
    rows = [r for j in plan["jobs"] for r in committed(output / f"audit/{j['id']}.json")["rows"]]
    groups = defaultdict(list)
    for r in rows: groups[(r["layout_id"], r["method"], r["delta"])].append(r)
    summary = {"scope": "exploratory_leave_one_low_cv_layout_out", "layout_ids": plan["cv_layout_ids"],
        "training_seed": plan["settings"]["cv_seed"], "fixed_epochs": plan["settings"]["cv_epochs"],
        "new_fits": 2*len(plan["cv_layout_ids"]), "new_pso_evaluations": 0,
        "main_deployment_models_changed": False, "held_out_used_for_checkpoint_or_delta_selection": False,
        "rows": len(rows), "groups": [{"layout_id": lid, "method": method, "delta": delta, **metrics(part)}
            for (lid, method, delta), part in sorted(groups.items())], "reserved_test_generated": False,
        "note": "Four layouts were identified after inspecting development data; this is exploratory and cannot replace the locked new-layout final test"}
    table(output / "metrics.csv", summary["groups"])
    table(output / "physics_rows.csv", rows)
    write(output / "summary.json", summary)
    paths = [p for p in output.rglob("*") if p.is_file() and p.name != "completion.json" and not p.name.endswith((".tmp", ".lock"))]
    freeze(output, "completion.json", paths, plan_sha256=digest(output / "plan.json"))
    return summary
