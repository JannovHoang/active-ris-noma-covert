"""Step 6c: compare paired PSO searches, then audit their frozen results."""
from collections import Counter
from copy import deepcopy
from itertools import groupby
from pathlib import Path
import platform
import time

from filelock import FileLock
import numpy as np
import scipy
import torch

from .benchmark_pso import (
    CODE_FILES as PSO_CODE, METHODS, commit_record, load_record, predict_start,
    settings_for, summarize_group, validate_config_5c, warm_positions, write_csv,
)
from .channels import build_bank, public_order
from .expanded_data import make_bank, save_json
from .expanded_learning import CODE_FILES as TRAINING_CODE, check_output, validate_source
from .expanded_scenarios import ROOT, physical_config, reproduce_layout
from .graph_data import original_allocation, statistical_graph
from .learning_data import TARGET_KEY, file_fingerprints, load_split, read_json, sha256
from .learning_physics import assess_allocation
from .learning_train import configure_runtime, load_checkpoint, predict, regression_metrics
from .objective import CachedObjective, decode_k2
from .pso import optimize


STREAMS = {"optimizer": 20, "warm_jitter": 21, "audit": 22}
CODE_FILES = tuple(dict.fromkeys((
    "run_step6c.py", "ris_noma/expanded_benchmark.py", *PSO_CODE, *TRAINING_CODE,
)))
TIMING_SCOPE = "per run with shared banks and objective cache already prepared"


def source_hashes(dataset, training):
    return {
        "source_6a_completion": sha256(dataset / "completion.json"),
        "source_6a_plan": sha256(dataset / "plan.json"),
        "source_6a_graphs": file_fingerprints(dataset / "graphs"),
        "source_6b_completion": sha256(training / "completion.json"),
        "source_6b_plan": sha256(training / "training_plan.json"),
        "source_6b_status": sha256(training / "training_status.json"),
    }


def select_models(dataset, training, hashes, bounds, schema):
    summary = check_output(training)
    plan = read_json(training / "training_plan.json")
    status = read_json(training / "training_status.json")
    if (plan["source_completion_sha256"] != hashes["source_6a_completion"]
            or plan["source_plan_sha256"] != hashes["source_6a_plan"]
            or plan["dataset_sha256"] != hashes["source_6a_graphs"]):
        raise ValueError("Step 6b checkpoints require the exact step 6a data used in training")
    if (plan["selection_metric"] != "validation_allocation_mse" or plan["test_used_for_selection"]
            or not status["training_complete"] or status["test_used_for_training_or_selection"]):
        raise ValueError("Need completed training with validation-only checkpoint selection")
    expected = {(kind, seed) for kind in ("gnn", "mlp") for seed in plan["config"]["seeds"]}
    runs = {(r["model"], r["seed"]): r for r in status["runs"]}
    if set(runs) != expected or len(status["runs"]) != len(expected):
        raise ValueError("Need all declared GNN and MLP checkpoints")
    for (kind, seed), row in runs.items():
        name = f"{kind}_seed_{seed}"
        if (row["name"] != name or row["test_used_for_selection"] or not np.isfinite(row["validation"]["mse"])
                or row["checkpoint_sha256"] != status["checkpoint_sha256"][name]
                or sha256(training / "training" / name / "best_model.pt") != row["checkpoint_sha256"]):
            raise ValueError(f"Checkpoint provenance mismatch: {name}")

    validation = load_split(dataset / "graphs", "validation")
    normalization = read_json(dataset / "graphs" / "normalization.json")
    selected, models = {}, {}
    for kind in ("gnn", "mlp"):
        best = min((r for r in runs.values() if r["model"] == kind),
                   key=lambda r: (r["validation"]["mse"], r["seed"]))
        name = best["name"]
        if name != status["selected_by_validation"][kind] or name != summary["selected_by_validation"][kind]:
            raise ValueError(f"Recorded {kind} selection differs from the minimum validation MSE")
        relative = f"training/{name}/best_model.pt"
        model, checkpoint = load_checkpoint(training / relative)
        mse = regression_metrics(predict(model, validation), validation[TARGET_KEY])["mse"]
        if (checkpoint["dataset_sha256"] != hashes["source_6a_graphs"]
                or checkpoint["normalization"] != normalization or checkpoint["bounds"] != bounds
                or checkpoint["feature_schema"] != schema or checkpoint["config"] != plan["config"]
                or (checkpoint["model_kind"], checkpoint["seed"], checkpoint["best_epoch"])
                != (kind, best["seed"], best["best_epoch"])
                or not np.isclose(mse, best["validation"]["mse"], rtol=1e-5, atol=1e-7)):
            raise ValueError(f"Frozen checkpoint does not reproduce validation: {name}")
        selected[kind] = {"name": name, "path": relative, "sha256": best["checkpoint_sha256"],
                          "training_seed": best["seed"], "best_epoch": best["best_epoch"],
                          "validation_mse": best["validation"]["mse"]}
        models[kind] = model, checkpoint
    return selected, models


def inspect_benchmark(dataset, training, config):
    c = validate_config_5c(deepcopy(config))
    configure_runtime(0, c["threads"], "cpu")
    _, schema = validate_source(dataset)
    source_plan = read_json(dataset / "plan.json")
    base = source_plan["base_config"]
    powers = source_plan["experiment"]["total_power_dbm"]
    xis = base["requirements"]["xi"]
    if any(xi + c["search_dep_margin"] >= 1 for xi in xis):
        raise ValueError("Every DEP target plus the search margin must be below one")
    hashes = source_hashes(dataset, training)
    selected, models = select_models(dataset, training, hashes, base["bounds"], schema)

    cases, split_counts = [], {}
    for role in c["splits"]:
        data = load_split(dataset / "graphs", role)
        expected = {(lid, power, xi) for lid in source_plan["partition"]["splits"][role]
                    for power in powers for xi in xis}
        keys = list(zip(data["layout_id"], data["total_power_dbm"], data["xi"]))
        if not expected or set(keys) != expected or len(keys) != len(expected):
            raise ValueError(f"Need complete multi-power pilot coverage in {role}")
        for (lid, power, xi), order in zip(keys, data["sic_order_original_ids"]):
            cases.append({"split": role, "layout_id": int(lid), "total_power_dbm": float(power), "xi": float(xi),
                          "power_index": powers.index(power), "xi_index": xis.index(xi),
                          "sic_order_original_ids": order.tolist()})
        split_counts[role] = len(keys)
    cases.sort(key=lambda case: (case["layout_id"], case["power_index"], case["xi_index"]))
    replicates = len(cases) * len(c["pso_seeds"])
    evaluations = sum(settings_for(c, method).swarm_size * settings_for(c, method).sweeps for method in METHODS)
    plan = {
        "step": "6c", "scope": "development_pilot", "config": c, "sources_sha256": hashes,
        "selected_checkpoints": selected, "checkpoint_selection": "6b validation MSE",
        "split_counts": split_counts, "cases": cases, "search_runs_planned": replicates * len(METHODS),
        "pso_evaluations_planned": replicates * evaluations, "streams": STREAMS,
        "search_bank_samples_per_layout": source_plan["experiment"]["label_samples"],
        "audit_family": "all thresholds, methods and optimizer seeds of the same layout and power",
        "timing_scope": TIMING_SCOPE, "code_sha256": {name: sha256(ROOT / name) for name in CODE_FILES},
        "versions": {"system": platform.system(), "machine": platform.machine(),
                     "python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "torch": str(torch.__version__)},
    }
    return plan, source_plan, models


def run_key(case, seed, method):
    return (f"layout_{case['layout_id']:06d}/power_{case['power_index']:03d}/"
            f"xi_{case['xi_index']:03d}/seed_{seed}/{method}.json")


def paired_rng(root_seed, case, stream, seed):
    entropy = [root_seed, case["layout_id"], STREAMS[stream], seed, case["power_index"], case["xi_index"]]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def search_one(c, case, seed, method, models, objective, bank, bank_seconds, cache_seconds):
    physical, order = objective.c, objective.order
    started = time.perf_counter()
    feature_seconds = inference_seconds = 0.
    initial = y_start = None
    if method in ("gnn_pso_short", "mlp_pso_short"):
        kind = method.split("_", 1)[0]
        tick = time.perf_counter()
        graph = statistical_graph(physical, objective.layout, bank, order, case["xi"])
        feature_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        y_start, center = predict_start(*models[kind], graph, order, physical["bounds"])
        initial = warm_positions(center, c["warm_particles"], c["warm_sigma"],
                                 paired_rng(physical["root_seed"], case, "warm_jitter", seed))
        inference_seconds = time.perf_counter() - tick
    settings = settings_for(c, method)
    target = case["xi"] + c["search_dep_margin"]
    tick = time.perf_counter()
    result = optimize(lambda z: objective.evaluate(z, target), 3, settings,
                      paired_rng(physical["root_seed"], case, "optimizer", seed), initial)
    pso_seconds = time.perf_counter() - tick
    mb, public, s = decode_k2(result.position, physical["bounds"])
    allocation, _ = original_allocation({"mb": mb, "public": public.tolist(), "s": s}, order, physical["bounds"])
    return {
        **case, "method": method, "pso_seed": seed, "search_target": target,
        "position_pso_sic": result.position.tolist(), "allocation_original_order": allocation.tolist(),
        "network_start_original_order": y_start.tolist() if y_start is not None else None,
        "evaluations": result.evaluations, "label_feasible": result.outcome.feasible,
        "label_metrics": result.outcome.metrics, "history": result.history,
        "timing": {"search_bank_setup_seconds": bank_seconds, "objective_setup_seconds": cache_seconds,
                   "feature_seconds": feature_seconds, "inference_and_warm_start_seconds": inference_seconds,
                   "pso_seconds": pso_seconds, "search_wall_seconds": time.perf_counter() - started},
    }


def search_layout(plan, source_plan, entry, cases, models, output):
    c = plan["config"]
    records = {}
    bank = None
    frozen = (output / "search_frozen.json").exists()
    for _, power_cases in groupby(cases, key=lambda case: case["power_index"]):
        objective = None
        for case in power_cases:
            order = np.asarray(case["sic_order_original_ids"], int)
            for index, seed in enumerate(c["pso_seeds"]):
                rotation = (case["layout_id"] + case["power_index"] + case["xi_index"] + index) % len(METHODS)
                for method in METHODS[rotation:] + METHODS[:rotation]:
                    key = run_key(case, seed, method)
                    path = output / "search" / key
                    record = load_record(path)
                    if record is None:
                        if frozen:
                            raise ValueError(f"Frozen search is missing: {key}")
                        if bank is None:
                            started = time.perf_counter()
                            base, layout, bank = make_bank(source_plan, entry, "label")
                            bank_seconds = time.perf_counter() - started
                            bank_order = public_order(bank)
                        if objective is None:
                            started = time.perf_counter()
                            physical = dict(base, total_dbm=case["total_power_dbm"])
                            objective = CachedObjective(physical, layout, bank, order)
                            cache_seconds = time.perf_counter() - started
                        if not objective.layout_eligible or not np.array_equal(order, bank_order):
                            raise ValueError("Search context differs from the eligible source layout")
                        record = search_one(c, case, seed, method, models, objective, bank, bank_seconds, cache_seconds)
                        commit_record(path, record)
                    records[key] = record
    return records


def audit_one(record, physical, layout, bank, bank_seconds, family_size, alpha, frozen_hash):
    order = np.asarray(record["sic_order_original_ids"], int)
    started = time.perf_counter()
    summary, screen = assess_allocation(physical, layout, bank, order, record["allocation_original_order"],
                                        record["xi"], family_size, alpha)
    audit_seconds = time.perf_counter() - started
    timing = record["timing"]
    search_seconds = timing["search_wall_seconds"]
    rate, passed = summary["bob_mean_rate_bit_s_hz"], screen["accepted"]
    qos = np.asarray(physical["requirements"]["qos_public"])[order]
    first = next((row["evaluations"] for row in record["history"] if row["gbest_feasible"]), None)
    return {
        **{key: record[key] for key in ("method", "split", "layout_id", "total_power_dbm", "xi", "pso_seed",
                                       "evaluations", "label_feasible", "allocation_original_order")},
        "bound_pass": passed, "bob_rate": rate, "audit_gated_bob_rate": rate if passed else 0.,
        "first_feasible_search_evaluation": first, "initial_swarm_feasible": record["history"][0]["gbest_feasible"],
        "dep_gamma": summary["dep_min"], "dep_empirical": screen["dep_empirical_min"],
        "dep_lower": screen["dep_uniform_lower"], "dep_lower_margin": screen["dep_lower_margin"],
        "min_qos_lower_margin": float(np.min(np.asarray(screen["public_rate_lower_sic_order"]) - qos)),
        "outage_upper": screen["bob_outage_upper"],
        "outage_upper_margin": physical["requirements"]["outage_bob_max"] - screen["bob_outage_upper"],
        "failed_checks": [key for key, ok in screen["checks"].items() if not ok],
        **timing, "audit_bank_setup_seconds": bank_seconds, "audit_wall_seconds": audit_seconds,
        "total_decision_seconds": search_seconds + audit_seconds,
        "standalone_estimate_seconds": search_seconds + audit_seconds + bank_seconds
        + timing["search_bank_setup_seconds"] + timing["objective_setup_seconds"],
        "screen": screen, "search_frozen_sha256": frozen_hash,
    }


def audit_layout(plan, source_plan, entry, records, families, frozen_hash, output):
    c = plan["config"]
    alpha = c["audit_alpha"] / len(families)
    bank = None
    rows = []
    for key, record in records:
        path = output / "audit" / key
        audit = load_record(path)
        power = record["total_power_dbm"]
        if audit is None:
            if bank is None:
                started = time.perf_counter()
                base = physical_config(source_plan, entry, power, c["audit_samples"])
                layout = reproduce_layout(base, entry)
                entropy = [base["root_seed"], entry["layout_id"], STREAMS["audit"], c["audit_seed"]]
                bank = build_bank(base, layout, np.random.default_rng(np.random.SeedSequence(entropy)))
                bank_seconds = time.perf_counter() - started
            audit = audit_one(record, dict(base, total_dbm=power), layout, bank, bank_seconds,
                              families[(entry["layout_id"], power)], alpha, frozen_hash)
            commit_record(path, audit)
        if audit["search_frozen_sha256"] != frozen_hash:
            raise ValueError(f"Audit belongs to a different frozen search: {key}")
        rows.append(audit)
    return rows


def group_summary(rows, split="all", power="all", xi="all"):
    table = []
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        stats = summarize_group(group, method)
        stats.update(split=split, total_power_dbm=power, xi=xi,
                     initial_swarm_pass_count=sum(row["initial_swarm_feasible"] for row in group),
                     standalone_seconds_mean=float(np.mean([row["standalone_estimate_seconds"] for row in group])))
        table.append(stats)
    full = table[0]
    for row in table:
        row["search_speedup_vs_full"] = full["search_seconds_mean"] / row["search_seconds_mean"]
        row["search_plus_audit_speedup_vs_full"] = full["total_seconds_mean"] / row["total_seconds_mean"]
        row["standalone_speedup_vs_full_estimate"] = full["standalone_seconds_mean"] / row["standalone_seconds_mean"]
    return table


def paired_summary(rows, split):
    pairs = {}
    for row in rows:
        key = tuple(row[field] for field in ("layout_id", "total_power_dbm", "xi", "pso_seed"))
        pairs.setdefault(key, {})[row["method"]] = row
    for method in ("gnn_pso_short", "mlp_pso_short"):
        matches = [(pair[method], pair["pso_short"]) for pair in pairs.values()]
        both = [(a, b) for a, b in matches if a["bound_pass"] and b["bound_pass"]]
        yield {
            "split": split, "method": method, "pairs": len(matches), "both_pass": len(both),
            "warm_only_pass": sum(a["bound_pass"] and not b["bound_pass"] for a, b in matches),
            "cold_only_pass": sum(b["bound_pass"] and not a["bound_pass"] for a, b in matches),
            "bob_rate_gain_both_pass_mean": float(np.mean([a["bob_rate"] - b["bob_rate"] for a, b in both])) if both else None,
            "gated_rate_gain_mean": float(np.mean([a["audit_gated_bob_rate"] - b["audit_gated_bob_rate"] for a, b in matches])),
        }


def summarize(output, plan, rows, families):
    by_method = group_summary(rows)
    by_split, by_power_dep, paired = [], [], list(paired_summary(rows, "all"))
    for split in plan["config"]["splits"]:
        group = [row for row in rows if row["split"] == split]
        by_split.extend(group_summary(group, split))
        paired.extend(paired_summary(group, split))
        for (power, xi), cases in groupby(sorted(group, key=lambda row: (row["total_power_dbm"], row["xi"])),
                                          key=lambda row: (row["total_power_dbm"], row["xi"])):
            by_power_dep.extend(group_summary(list(cases), split, power, xi))
    flat = [{key: value for key, value in row.items() if key != "screen"} for row in rows]
    for name, table in (("runs", flat), ("by_method", by_method), ("by_split", by_split),
                        ("by_power_dep", by_power_dep), ("paired_vs_short", paired)):
        write_csv(output / f"comparison_{name}.csv", table)
    c = plan["config"]
    return {
        "step": "6c", "scope": "development_pilot", "benchmark_complete": True,
        "search_runs": len(rows), "total_pso_evaluations": sum(row["evaluations"] for row in rows),
        "split_counts": plan["split_counts"], "selected_checkpoints": plan["selected_checkpoints"],
        "audit": {"samples_per_layout": c["audit_samples"], "seed": c["audit_seed"], "stream": STREAMS["audit"],
                  "layout_power_families": len(families), "family_sizes": sorted(set(families.values())),
                  "alpha_total": c["audit_alpha"], "alpha_per_family": c["audit_alpha"] / len(families)},
        "timing_scope": TIMING_SCOPE, "summary": by_method, "by_split": by_split, "paired_vs_short": paired,
    }


def check_benchmark(output):
    output = Path(output).resolve()
    files = read_json(output / "completion.json")["sha256"]
    required = {"benchmark_plan.json", "search_frozen.json", "benchmark_status.json", "benchmark_summary.json",
                "comparison_runs.csv", "comparison_by_method.csv", "comparison_by_split.csv",
                "comparison_by_power_dep.csv", "comparison_paired_vs_short.csv"}
    if not required.issubset(files):
        raise ValueError("Incomplete step 6c numerical output")
    for name, digest in files.items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Step 6c result changed: {name}")
    result = read_json(output / "benchmark_summary.json")
    plan = read_json(output / "benchmark_plan.json")
    frozen = read_json(output / "search_frozen.json")["search_records_sha256"]
    if (result["step"] != "6c" or not result["benchmark_complete"]
            or len(frozen) != result["search_runs"] or len(frozen) != plan["search_runs_planned"]
            or result["total_pso_evaluations"] != plan["pso_evaluations_planned"]
            or any(files.get(f"search/{key}") != digest or f"audit/{key}" not in files for key, digest in frozen.items())):
        raise ValueError("Step 6c numerical output does not match the declared plan")
    return result


def run_benchmark(dataset, training, output, config, resume=False):
    dataset, training, output = map(lambda path: Path(path).resolve(), (dataset, training, output))
    for source in (dataset, training):
        if source == output or source in output.parents or output in source.parents:
            raise ValueError("Input and output must be separate, non-nested directories")
    plan, source_plan, models = inspect_benchmark(dataset, training, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output.with_name(f".{output.name}.lock")), timeout=0):
        return run_locked(dataset, training, output, plan, source_plan, models, resume)


def run_locked(dataset, training, output, plan, source_plan, models, resume):
    plan_path = output / "benchmark_plan.json"
    if output.exists() and any(output.iterdir()):
        if not resume or not plan_path.is_file() or read_json(plan_path) != plan:
            raise ValueError("Use a new output or --resume with the identical benchmark plan")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "completion.json").is_file():
        return check_benchmark(output)
    if not plan_path.is_file():
        save_json(plan_path, plan)

    entries = {row["layout_id"]: row for row in source_plan["layouts"]}
    searches = {}
    for lid, cases in groupby(plan["cases"], key=lambda case: case["layout_id"]):
        searches.update(search_layout(plan, source_plan, entries[lid], cases, models, output))
        print(f"Search layout {lid}: {len(searches)}/{plan['search_runs_planned']} runs committed", flush=True)
    if (len(searches) != plan["search_runs_planned"]
            or sum(row["evaluations"] for row in searches.values()) != plan["pso_evaluations_planned"]):
        raise ValueError("Search counts differ from the paired benchmark plan")
    frozen = {"search_records_sha256": {key: sha256(output / "search" / key) for key in sorted(searches)},
              "count": len(searches), "benchmark_plan_sha256": sha256(plan_path)}
    frozen_path = output / "search_frozen.json"
    if frozen_path.exists() and read_json(frozen_path) != frozen:
        raise ValueError("Frozen searches changed")
    if not frozen_path.exists():
        save_json(frozen_path, frozen)
    frozen_hash = sha256(frozen_path)
    print(f"All {len(searches)} searches frozen. Starting the independent fading audit.", flush=True)

    families = Counter((r["layout_id"], r["total_power_dbm"]) for r in searches.values())
    rows = []
    for lid, group in groupby(sorted(searches.items()), key=lambda pair: pair[1]["layout_id"]):
        rows.extend(audit_layout(plan, source_plan, entries[lid], group, families, frozen_hash, output))
        print(f"Audit layout {lid}: {len(rows)}/{len(searches)} runs committed", flush=True)
    if (sha256(frozen_path) != frozen_hash or source_hashes(dataset, training) != plan["sources_sha256"]
            or any(sha256(training / row["path"]) != row["sha256"] for row in plan["selected_checkpoints"].values())
            or any(sha256(output / "search" / key) != digest for key, digest in frozen["search_records_sha256"].items())):
        raise ValueError("Source data, selected checkpoints or frozen searches changed during step 6c")
    result = summarize(output, plan, rows, families)
    save_json(output / "benchmark_summary.json", result)
    save_json(output / "benchmark_status.json", {"benchmark_complete": True, "runs": len(rows),
                                                "search_frozen_sha256": frozen_hash})
    save_json(output / "completion.json", {"sha256": {
        path.relative_to(output).as_posix(): sha256(path) for path in sorted(output.rglob("*"))
        if path.is_file() and path.suffix != ".tmp" and path != output / "completion.json"
        and (output / "figures") not in path.parents
    }})
    return check_benchmark(output)
