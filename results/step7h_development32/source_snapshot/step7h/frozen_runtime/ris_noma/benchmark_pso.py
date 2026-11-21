"""Step 5c: paired PSO budgets with frozen validation-selected NN starts.

No old PSO allocation is supplied to search. Every search finishes and is
hashed before a fresh audit bank is constructed. Previously viewed pilot
layouts are developmental data, even when their 5a split is named 'test'.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import csv
import json
import math
import platform
import time
import numpy as np
import torch
from .config import validate_config
from .geometry import sample_layout
from .channels import build_bank, public_order
from .random_streams import make_rng
from .objective import CachedObjective, decode_k2
from .pso import PSOSettings, optimize
from .graph_data import statistical_graph, transform_graphs, original_allocation
from .learning_data import (inspect_export, load_split, read_json, sha256,
                            file_fingerprints, physics_allocation, TARGET_KEY)
from .learning_train import load_checkpoint, configure_runtime, predict, regression_metrics
from .learning_physics import regenerate_label_context, assess_allocation

METHODS = ("pso_full", "pso_short", "gnn_pso_short", "mlp_pso_short")
STREAMS = {"audit": 11, "optimizer": 12, "warm_jitter": 13}
ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = ["run_step5c.py", "ris_noma/benchmark_pso.py", "ris_noma/benchmark_plots.py",
              "ris_noma/pso.py", "ris_noma/objective.py", "ris_noma/evaluator.py", "ris_noma/metrics.py",
              "ris_noma/dep.py", "ris_noma/channels.py", "ris_noma/geometry.py", "ris_noma/random_streams.py",
              "ris_noma/validation.py", "ris_noma/graph_data.py", "ris_noma/learning_models.py",
              "ris_noma/learning_train.py", "ris_noma/learning_physics.py", "ris_noma/learning_data.py"]


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def commit_record(path, value):
    atomic_json(path, value)
    atomic_json(str(path) + ".receipt.json", {"sha256": sha256(path)})


def load_record(path):
    path = Path(path)
    receipt = Path(str(path) + ".receipt.json")
    if not receipt.exists():
        return None  # An interrupted, uncommitted record can be recomputed.
    if not path.exists() or read_json(receipt)["sha256"] != sha256(path):
        raise ValueError(f"Changed or incomplete committed result: {path}")
    return read_json(path)


def validate_config_5c(c):
    names = {"purpose", "splits", "methods", "pso_seeds", "swarm_size", "full_sweeps", "short_sweeps",
             "inertia_start", "inertia_end", "cognitive", "social", "velocity_max", "initial_velocity_scale",
             "warm_particles", "warm_sigma", "search_dep_margin", "audit_samples", "audit_seed", "audit_alpha", "threads", "notes"}
    if set(c) != names or c["purpose"] != "development_pilot":
        raise ValueError("Need the declared development_pilot benchmark configuration")
    if not c["splits"] or len(set(c["splits"])) != len(c["splits"]) or any(s not in ["validation", "test"] for s in c["splits"]):
        raise ValueError("Benchmark only predeclared validation/test pilot layouts")
    if set(c["methods"]) != set(METHODS) or len(c["methods"]) != len(METHODS):
        raise ValueError("Declare all four comparison methods exactly once")
    if not c["pso_seeds"] or len(set(c["pso_seeds"])) != len(c["pso_seeds"]) or any(type(s) is not int or s < 0 for s in c["pso_seeds"]):
        raise ValueError("PSO seeds must be distinct nonnegative integers")
    for key in ["swarm_size", "full_sweeps", "short_sweeps", "warm_particles", "audit_samples", "threads"]:
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not 1 <= c["warm_particles"] < c["swarm_size"] or c["audit_samples"] < 2 or c["short_sweeps"] >= c["full_sweeps"]:
        raise ValueError("Keep random particles; need fewer short than full sweeps and at least 2 audit samples")
    for key in ["warm_sigma", "search_dep_margin"]:
        if not np.isfinite(c[key]) or c[key] < 0:
            raise ValueError(f"Invalid {key}")
    if type(c["audit_seed"]) is not int or c["audit_seed"] < 0 or not 0 < c["audit_alpha"] < 1:
        raise ValueError("Invalid audit seed/alpha")
    settings_for(c, "pso_full"); settings_for(c, "pso_short")
    return c


def settings_for(c, method):
    names = ["swarm_size", "inertia_start", "inertia_end", "cognitive", "social", "velocity_max", "initial_velocity_scale"]
    return PSOSettings(**{n: c[n] for n in names}, sweeps=c["full_sweeps"] if method == "pso_full" else c["short_sweeps"])


def role_rng(root_seed, layout_id, role, seed, xi_index=0):
    return np.random.default_rng(np.random.SeedSequence([root_seed, layout_id, STREAMS[role], seed, xi_index]))


def warm_positions(center, count, sigma, rng):
    center = np.asarray(center, float)
    if center.shape != (3,) or not np.all(np.isfinite(center)) or np.any((center < 0) | (center > 1)) or count < 1 or sigma < 0:
        raise ValueError("Invalid normalized warm start")
    result = np.repeat(center[None], count, axis=0)
    if count > 1:
        result[1:] = np.clip(center + rng.normal(0., sigma, size=(count-1, 3)), 0., 1.)
    return result


def select_checkpoints(training, dataset, c):
    training, dataset = Path(training), Path(dataset)
    status, source_plan = read_json(training / "training_status.json"), read_json(training / "training_plan.json")
    if not status["training_complete"] or not status["all_declared_seeds_reported"]:
        raise ValueError("Need all completed step-5b training runs")
    if source_plan["test_used_for_training_or_checkpoint_selection"] or any(r["test_used_for_selection"] for r in status["runs"]):
        raise ValueError("Source training used test for selection")
    fingerprint = file_fingerprints(dataset)
    if source_plan["dataset_sha256"] != fingerprint:
        raise ValueError("Dataset differs from the one used by the uploaded checkpoints")
    for name, digest in status["checkpoint_sha256"].items():
        if sha256(training / name) != digest:
            raise ValueError(f"Source checkpoint changed: {name}")
    selected, models, setup = {}, {}, {}
    validation = load_split(dataset, "validation")
    for kind in ["gnn", "mlp"]:
        candidates = [r for r in status["runs"] if r["model"] == kind]
        if not candidates or any(not np.isfinite(r["validation"]["mse"]) for r in candidates):
            raise ValueError("Need finite validation MSE for both architectures")
        best = min(candidates, key=lambda r: (r["validation"]["mse"], r["seed"]))
        relative = f"{kind}_seed_{best['seed']}/best_model.pt"
        before = time.perf_counter()
        model, checkpoint = load_checkpoint(training / relative, "cpu")
        actual = regression_metrics(predict(model, validation), validation[TARGET_KEY])["mse"]
        if checkpoint["dataset_sha256"] != fingerprint or not np.isclose(actual, best["validation"]["mse"], rtol=1e-5, atol=1e-7):
            raise ValueError("Selected checkpoint does not reproduce its validation metric")
        setup[kind + "_load_and_validation_check_seconds"] = time.perf_counter() - before
        selected[kind] = {"path": relative, "sha256": sha256(training / relative), "training_seed": best["seed"],
                          "best_epoch": best["best_epoch"], "validation_mse": best["validation"]["mse"],
                          "choice": "lowest recorded validation MSE; ties by training seed", "test_used_for_choice": False}
        models[kind] = (model, checkpoint)
    return selected, models, setup


def inspect_benchmark(dataset, training, config):
    c = validate_config_5c(deepcopy(config))
    configure_runtime(0, c["threads"], "cpu")
    inspection = inspect_export(dataset)
    if not inspection["ready_for_training_pilot"]:
        raise ValueError("Need a complete pilot graph export")
    manifest = read_json(Path(dataset) / "dataset_manifest.json")
    if any(x + c["search_dep_margin"] >= 1 for x in manifest["base_physics_config"]["requirements"]["xi"]):
        raise ValueError("DEP target plus the fixed search margin must be below 1")
    selected, models, setup = select_checkpoints(training, dataset, c)
    cases = []
    for split in c["splits"]:
        d = load_split(dataset, split)
        for lid, xi, order in zip(d["layout_id"], d["xi"], d["sic_order_original_ids"]):
            cases.append({"split": split, "layout_id": int(lid), "xi": float(xi), "sic_order_original_ids": order.tolist()})
    if not cases:
        raise ValueError("No benchmark cases")
    budget = len(cases) * len(c["pso_seeds"]) * sum(settings_for(c, m).swarm_size * settings_for(c, m).sweeps for m in METHODS)
    info = {"cases": cases, "selected_checkpoints": selected, "search_runs": len(cases)*len(c["pso_seeds"])*4,
            "pso_evaluations_planned": budget, "input_files_sha256": inspection["input_files_sha256"],
            "scope": "previously viewed pilot layouts; no final independent generalization or speedup claim"}
    return c, manifest, info, models, setup


def predict_start(model, checkpoint, graph, order, bounds):
    """Only pre-allocation graph inputs; no target allocation is accepted."""
    node, glob = transform_graphs(graph["node_raw"][None], graph["global_raw"][None], checkpoint["normalization"])
    inputs = {"node": torch.tensor(node, dtype=torch.float32), "global_features": torch.tensor(glob, dtype=torch.float32),
              "edge_index": torch.tensor(graph["edge_index"][None], dtype=torch.long),
              "edge_attr": torch.tensor(graph["edge_attr"][None], dtype=torch.float32)}
    with torch.inference_mode():
        y = model(**inputs).cpu().numpy()[0]
    y = physics_allocation(y, bounds)
    _, z = original_allocation({"mb": y[0], "public": y[1:3][order].tolist(), "s": y[3]}, order, bounds)
    return y, z


def run_key(case, seed, method):
    xi = int(round(case["xi"] * 1000000))
    return f"layout_{case['layout_id']:06d}/xi_{xi:06d}/seed_{seed}/{method}.json"


def search_one(base, case, seed, method, c, models, xi_index):
    start = time.perf_counter()
    physical = validate_config(deepcopy(base))
    physical["layout_id"] = case["layout_id"]
    layout = sample_layout(physical, make_rng(physical["root_seed"], case["layout_id"], "geometry"))
    bank = build_bank(physical, layout, make_rng(physical["root_seed"], case["layout_id"], "label"))
    order = np.asarray(case["sic_order_original_ids"], int)
    if not np.array_equal(order, public_order(bank)):
        raise ValueError("Frozen source SIC order no longer matches the label bank")
    objective = CachedObjective(physical, layout, bank, order)
    if not objective.layout_eligible:
        raise ValueError("Layout eligibility differs from the source label rule")
    prepare_seconds = time.perf_counter() - start
    feature_seconds, inference_seconds = 0., 0.
    initial, y_start = None, None
    if method in ["gnn_pso_short", "mlp_pso_short"]:
        kind = "gnn" if method.startswith("gnn") else "mlp"
        tick = time.perf_counter()
        graph = statistical_graph(physical, layout, bank, order, case["xi"])
        feature_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        y, center = predict_start(*models[kind], graph, order, physical["bounds"])
        initial = warm_positions(center, c["warm_particles"], c["warm_sigma"],
                                 role_rng(physical["root_seed"], case["layout_id"], "warm_jitter", seed, xi_index))
        inference_seconds = time.perf_counter() - tick
        y_start = y.tolist()
    settings = settings_for(c, method)
    target = case["xi"] + c["search_dep_margin"]
    tick = time.perf_counter()
    result = optimize(lambda z: objective.evaluate(z, target), 3, settings,
                      role_rng(physical["root_seed"], case["layout_id"], "optimizer", seed, xi_index), initial)
    optimization_seconds = time.perf_counter() - tick
    elapsed = time.perf_counter() - start
    mb, mp, s = decode_k2(result.position, physical["bounds"])
    y, _ = original_allocation({"mb": mb, "public": mp.tolist(), "s": s}, order, physical["bounds"])
    record = {**case, "pso_seed": seed, "method": method, "allocation_original_order": y.tolist(),
              "position_pso_sic": result.position.tolist(), "network_start_original_order": y_start,
              "search_target": target, "search_settings": asdict(settings), "evaluations": result.evaluations,
              "label_feasible": result.outcome.feasible, "label_rate": result.outcome.score,
              "label_metrics": result.outcome.metrics, "history": result.history,
              "timing": {"prepare_search_bank_and_cache_seconds": prepare_seconds, "feature_seconds": feature_seconds,
                         "inference_and_warm_start_seconds": inference_seconds, "pso_seconds": optimization_seconds,
                         "search_wall_seconds": elapsed}, "source_label_allocation_used_for_start": False,
              "heldout_used_for_search": False}
    del objective, bank
    return record


def audit_one(base, record, c, family_size, n_layouts):
    start = time.perf_counter()
    physical = validate_config(deepcopy(base)); physical.update(layout_id=record["layout_id"], n_fading=c["audit_samples"])
    layout = sample_layout(physical, make_rng(physical["root_seed"], record["layout_id"], "geometry"))
    bank = build_bank(physical, layout, role_rng(physical["root_seed"], record["layout_id"], "audit", c["audit_seed"]))
    prepare_seconds = time.perf_counter() - start
    tick = time.perf_counter()
    summary, screen = assess_allocation(physical, layout, bank, np.asarray(record["sic_order_original_ids"]),
                                        record["allocation_original_order"], record["xi"], family_size, c["audit_alpha"]/n_layouts)
    assess_seconds = time.perf_counter() - tick
    audit_seconds = time.perf_counter() - start
    order = np.asarray(record["sic_order_original_ids"], int)
    qos = np.asarray(physical["requirements"]["qos_public"])[order]
    passed = screen["accepted"]
    result = {"method": record["method"], "split": record["split"], "layout_id": record["layout_id"],
              "xi": record["xi"], "pso_seed": record["pso_seed"], "evaluations": record["evaluations"],
              "label_feasible": record["label_feasible"], "bound_pass": passed,
              "gamma_point_feasible": summary["constraints"]["feasible_under_gamma_sample_model"],
              "bob_rate": summary["bob_mean_rate_bit_s_hz"],
              "audit_gated_bob_rate": summary["bob_mean_rate_bit_s_hz"] if passed else 0.,
              "dep_gamma": summary["dep_min"], "dep_empirical": screen["dep_empirical_min"],
              "dep_lower": screen["dep_uniform_lower"], "dep_lower_margin": screen["dep_lower_margin"],
              "min_qos_lower_margin": float(np.min(np.asarray(screen["public_rate_lower_sic_order"])-qos)),
              "outage_upper": screen["bob_outage_upper"],
              "outage_upper_margin": physical["requirements"]["outage_bob_max"]-screen["bob_outage_upper"],
              "failed_checks": [k for k,v in screen["checks"].items() if not v],
              "allocation_original_order": record["allocation_original_order"],
              "search_wall_seconds": record["timing"]["search_wall_seconds"],
              "audit_prepare_seconds": prepare_seconds, "audit_assessment_seconds": assess_seconds,
              "audit_wall_seconds": audit_seconds,
              "total_decision_seconds": record["timing"]["search_wall_seconds"]+audit_seconds,
              "timing_scope": "search plus fresh audit, each bank rebuilt per decision; excludes one-off loading, file I/O, plotting and offline training",
              "screen": screen}
    del bank
    return result


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def summarize_group(rows, method, split="all", xi=None):
    selected = [r for r in rows if r["method"]==method and (split=="all" or r["split"]==split) and (xi is None or r["xi"]==xi)]
    passed = [r for r in selected if r["bound_pass"]]
    def mean(key): return float(np.mean([r[key] for r in selected]))
    return {"method":method,"split":split,"xi":"all" if xi is None else xi,"runs":len(selected),
            "layouts":len({r["layout_id"] for r in selected}),"bound_pass_count":len(passed),
            "bound_pass_fraction":len(passed)/len(selected), "bob_rate_all_mean":mean("bob_rate"),
            "bob_rate_passing_only_mean":float(np.mean([r["bob_rate"] for r in passed])) if passed else None,
            "audit_gated_bob_rate_mean":mean("audit_gated_bob_rate"),
            "search_seconds_mean":mean("search_wall_seconds"),"audit_seconds_mean":mean("audit_wall_seconds"),
            "total_seconds_mean":mean("total_decision_seconds"),"evaluations_mean":mean("evaluations"),
            "min_dep_lower_margin":min(r["dep_lower_margin"] for r in selected)}


def summarize(output, plan, audits):
    output=Path(output)
    by_method=[summarize_group(audits,m) for m in METHODS]
    by_xi=[summarize_group(audits,m,xi=x) for x in sorted({r["xi"] for r in audits}) for m in METHODS]
    by_split=[summarize_group(audits,m,split=s) for s in plan["config"]["splits"] for m in METHODS]
    write_csv(output/"comparison_by_xi.csv",by_xi);write_csv(output/"comparison_by_split.csv",by_split)
    flat=[{k:v for k,v in r.items() if k not in ["screen","allocation_original_order","failed_checks"]}|
          {"failed_checks":";".join(r["failed_checks"])} for r in audits]
    write_csv(output/"comparison_runs.csv",flat)
    base=next(r for r in by_method if r["method"]=="pso_full")
    for r in by_method:
        r["ratio_of_mean_search_times_vs_full"]=base["search_seconds_mean"]/r["search_seconds_mean"]
        r["ratio_of_mean_total_times_vs_full"]=base["total_seconds_mean"]/r["total_seconds_mean"]
    write_csv(output/"comparison_by_method.csv",by_method)
    conclusion={"benchmark_complete":True,"purpose":"development_pilot","final_gnn_test":False,
                "search_runs":len(audits),"total_pso_evaluations":sum(r["evaluations"] for r in audits),
                "selected_checkpoints":plan["selected_checkpoints"],"summary":by_method,
                "scope":"Timing and quality on previously inspected pilot layouts; throughput speed alone is not a quality guarantee",
                "rate_gate_definition":"Bob rate is set to zero when any declared audit check fails; this is an accounting score, not physical goodput",
                "timing_note":"Each search and each audit bank rebuilt per run. Total is the sum of the two measured stages; resident model loading and training are separate setup costs.",
                "model_training_performed":False,"heldout_used_for_search_or_reselection":False,"fallback_runs":0}
    atomic_json(output/"benchmark_summary.json",conclusion)
    return conclusion


def run_benchmark(dataset, training, output, config, resume=False):
    dataset,training,output=Path(dataset).resolve(),Path(training).resolve(),Path(output).resolve()
    for source in [dataset,training]:
        if source==output or source in output.parents or output in source.parents:
            raise ValueError("Source and benchmark output must be separate, non-nested directories")
    c,manifest,info,models,setup=inspect_benchmark(dataset,training,config)
    base=deepcopy(manifest["base_physics_config"]);base["n_fading"]=manifest["label_bank_samples_per_layout"]
    plan={"step":"5c","config":c,"dataset_path":str(dataset),"training_path":str(training),
          "dataset_sha256":info["input_files_sha256"],"selected_checkpoints":info["selected_checkpoints"],
          "source_training_plan_sha256":sha256(training/"training_plan.json"),
          "source_training_status_sha256":sha256(training/"training_status.json"),
          "cases":info["cases"],"search_runs_planned":info["search_runs"],"pso_evaluations_planned":info["pso_evaluations_planned"],
          "streams":STREAMS,"base_config":base,"code_sha256":{n:sha256(ROOT/n) for n in CODE_FILES},
          "versions":{"python":platform.python_version(),"numpy":np.__version__,"torch":str(torch.__version__)},
          "checkpoint_choice_uses":"validation MSE only","final_gnn_test":False,
          "audit_family":"all methods x optimizer seeds x xi of the same layout, declared before search",
          "no_label_solution_initialization":True,"no_audit_reselection":True}
    if output.exists() and any(output.iterdir()):
        if not resume or not (output/"benchmark_plan.json").exists() or read_json(output/"benchmark_plan.json")!=plan:
            raise ValueError("Nonempty/changed benchmark; use --resume with the same plan or a new output")
    output.mkdir(parents=True,exist_ok=True)
    if not (output/"benchmark_plan.json").exists():
        atomic_json(output/"benchmark_plan.json",plan);atomic_json(output/"setup_measurements.json",setup)
    # Validate source reproduction once per layout, outside decision timings.
    for lid in sorted({r["layout_id"] for r in info["cases"]}):
        split=next(r["split"] for r in info["cases"] if r["layout_id"]==lid)
        regenerate_label_context(dataset,lid,split)
    searches=[]
    xis=base["requirements"]["xi"]
    for j,case in enumerate(info["cases"]):
        for k,seed in enumerate(c["pso_seeds"]):
            # Rotate execution order; pairing of optimizer RNG is unchanged.
            rotation=(j+k)%len(METHODS)
            methods=list(METHODS[rotation:]+METHODS[:rotation])
            for method in methods:
                key=run_key(case,seed,method);path=output/"search"/key
                record=load_record(path) if resume else None
                if record is None:
                    record=search_one(base,case,seed,method,c,models,xis.index(case["xi"]))
                    commit_record(path,record)
                searches.append((key,record))
        print(f"Search {j+1}/{len(info['cases'])}: layout={case['layout_id']} xi={case['xi']:.2f}; all 4 methods x {len(c['pso_seeds'])} seeds",flush=True)
    frozen={"search_records_sha256":{key:sha256(output/"search"/key) for key,_ in sorted(searches)},
            "count":len(searches),"heldout_generated_before_freeze":False}
    frozen_path=output/"search_frozen.json"
    if frozen_path.exists() and read_json(frozen_path)!=frozen:
        raise ValueError("Frozen searches changed; audit cannot continue")
    if not frozen_path.exists():atomic_json(frozen_path,frozen)
    frozen_hash=sha256(frozen_path)
    atomic_json(output/"search_status.json",{"search_complete":True,"runs":len(searches),"search_frozen_sha256":frozen_hash})
    print(f"All {len(searches)} searches frozen. Starting fresh independent fading audit.",flush=True)
    audits=[]
    n_layouts=len({r["layout_id"] for _,r in searches})
    for j,(key,record) in enumerate(searches):
        family_size=sum(r["layout_id"]==record["layout_id"] for _,r in searches)
        path=output/"audit"/key
        result=load_record(path) if resume else None
        if result is None:
            result=audit_one(base,record,c,family_size,n_layouts);result["search_frozen_sha256"]=frozen_hash
            commit_record(path,result)
        if result["search_frozen_sha256"]!=frozen_hash:
            raise ValueError("Audit does not belong to these frozen searches")
        audits.append(result)
        if (j+1)%12==0 or j+1==len(searches):print(f"Audit {j+1}/{len(searches)} complete",flush=True)
    if sha256(frozen_path)!=frozen_hash or file_fingerprints(dataset)!=plan["dataset_sha256"]:
        raise ValueError("Frozen selection or source data changed")
    for kind,selected in plan["selected_checkpoints"].items():
        if sha256(training/selected["path"])!=selected["sha256"]:raise ValueError("Checkpoint changed during benchmark")
    summary=summarize(output,plan,audits)
    atomic_json(output/"benchmark_status.json",{"benchmark_complete":True,"runs":len(audits),
                "search_frozen_sha256":frozen_hash,"source_data_and_selected_checkpoints_unchanged":True,"heldout_used_for_selection":False})
    return summary
