"""Freeze model selection and every search proposal before the fresh 6f test audit."""
from dataclasses import replace
from itertools import product
import time

import numpy as np

from .benchmark_pso import predict_start, settings_for, warm_positions
from .budget_benchmark import run_phase
from .channels import public_order
from .dep_label_benchmark import summarize_group
from .dep_label_experiment import channel_context, freeze_phase, rng
from .dep_label_learning import read_model, score_cases, write_immutable
from .expanded_data import write_table
from .graph_data import original_allocation, statistical_graph
from .learning_data import sha256
from .learning_train import configure_runtime
from .objective import CachedObjective, decode_k2
from .physics_training import GROUPS
from .pso import optimize


PSO_METHODS = ("pso_short", "baseline_pso_short", "selected_pso_short")


def search_layout(entry, plan, source, protocol, output, freeze):
    record = {"layout_id": entry["layout_id"], "selection_frozen_sha256": freeze,
              "status": "invalid_geometry", "rows": []}
    if not entry["geometry_valid"]:
        return record
    started = time.perf_counter()
    base, layout, bank = channel_context(plan, source, entry, "test_search", plan["search_samples_per_layout"])
    bank_seconds = time.perf_counter() - started
    order = public_order(bank)
    if not CachedObjective(base, layout, bank, order).layout_eligible:
        return dict(record, status="bob_not_strongest")
    record.update(status="eligible", order=order.tolist(), search_bank_seconds=bank_seconds)
    c = plan["search_config"]
    configure_runtime(0, c["threads"], "cpu")
    models = {name: read_model(ref, output) for name, ref in protocol["direct"].items()}
    settings = replace(settings_for(c, "pso_short"), sweeps=plan["benchmark_budget"] // c["swarm_size"])
    # One untimed forward pass per loaded model avoids charging kernel startup to one method.
    warmup_graph = statistical_graph(base, layout, bank, order, plan["xi"][0])
    for model in models.values():
        predict_start(*model, warmup_graph, order, base["bounds"])
    for power_idx, power in enumerate(plan["power_dbm"]):
        physical = dict(base, total_dbm=power)
        started = time.perf_counter()
        objective = CachedObjective(physical, layout, bank, order)
        cache_seconds = time.perf_counter() - started
        for xi_idx, xi in enumerate(plan["xi"]):
            case = {"layout_id": entry["layout_id"], "total_power_dbm": power, "xi": xi}
            names = list(models)
            rotation = (entry["layout_id"] + power_idx + xi_idx) % len(names)
            for method in names[rotation:] + names[:rotation]:
                started = time.perf_counter()
                graph = statistical_graph(physical, layout, bank, order, xi)
                y, _ = predict_start(*models[method], graph, order, physical["bounds"])
                record["rows"].append({**case, "method": method, "pso_seed": None, "evaluations": 0,
                    "allocation": y.tolist(), "search_wall_seconds": time.perf_counter() - started,
                    "search_bank_seconds": bank_seconds, "objective_cache_seconds": cache_seconds})
            for seed_idx, seed in enumerate(c["pso_seeds"]):
                rotation = (entry["layout_id"] + power_idx + xi_idx + seed_idx) % len(PSO_METHODS)
                for method in PSO_METHODS[rotation:] + PSO_METHODS[:rotation]:
                    started = time.perf_counter()
                    initial = None
                    if method in protocol["warm"]:
                        graph = statistical_graph(physical, layout, bank, order, xi)
                        _, center = predict_start(*models[protocol["warm"][method]], graph, order, physical["bounds"])
                        initial = warm_positions(center, c["warm_particles"], c["warm_sigma"],
                                                  rng(plan, base, entry, "warm_jitter", seed, power_idx, xi_idx))
                    result = optimize(lambda z: objective.evaluate(z, xi + c["search_dep_margin"]), 3, settings,
                                      rng(plan, base, entry, "test_pso", seed, power_idx, xi_idx), initial)
                    if result.evaluations != plan["benchmark_budget"]:
                        raise ArithmeticError("PSO comparisons must use exactly the same evaluation budget")
                    mb, public, s = decode_k2(result.position, physical["bounds"])
                    allocation, _ = original_allocation({"mb": mb, "public": public, "s": s}, order, physical["bounds"])
                    record["rows"].append({**case, "method": method, "pso_seed": seed,
                        "evaluations": result.evaluations, "allocation": allocation.tolist(),
                        "search_wall_seconds": time.perf_counter() - started,
                        "search_bank_seconds": bank_seconds, "objective_cache_seconds": cache_seconds})
    return record


def audit_layout(entry, plan, source, search, freeze):
    base, layout, bank = channel_context(plan, source, entry, "test_audit", plan["audit_samples_per_layout"])
    rows, unique = score_cases(base, layout, bank, np.asarray(search["order"], int), search["rows"],
                               plan["test_family_size"], plan["test_alpha_per_family"])
    return {"layout_id": entry["layout_id"], "test_search_frozen_sha256": freeze, "rows": rows, "unique": unique}


def paired_contrast(name, method_a, method_b, grouped, lids, bootstrap_indices):
    rows_a, rows_b = grouped[method_a], grouped[method_b]
    if not rows_a or not rows_b:
        raise ValueError("Need both methods for a paired contrast")
    result = {"contrast": name, "method_a": method_a, "method_b": method_b,
              "layouts": len(lids), "runs_a": len(rows_a), "runs_b": len(rows_b)}
    for field, factor, metric in (("audit_pass", 100., "pass_delta_pp"),
            ("gated_bob_rate", 1., "gated_rate_delta"), ("search_wall_seconds", 1000., "wall_time_delta_ms")):
        values = np.array([factor * (np.mean([r[field] for r in rows_a if r["layout_id"] == lid])
                                    - np.mean([r[field] for r in rows_b if r["layout_id"] == lid])) for lid in lids])
        result[metric] = float(values.mean())
        low, high = (None, None) if bootstrap_indices is None else np.quantile(values[bootstrap_indices].mean(1), [.025, .975]).tolist()
        result[f"{metric}_ci_low"], result[f"{metric}_ci_high"] = low, high
    return result


def aggregate(output, plan, protocol, rows):
    grouped = {method: [r for r in rows if r["method"] == method] for method in (*protocol["direct"], *PSO_METHODS)}
    for group in GROUPS:
        grouped[group] = [r for name, ref in protocol["direct"].items() if ref["group"] == group for r in grouped[name]]
    grouped["baseline_direct_selected"] = grouped[protocol["warm"]["baseline_pso_short"]]
    grouped["chosen_direct_selected"] = grouped[protocol["warm"]["selected_pso_short"]]
    table = [summarize_group(group, method) for method, group in grouped.items()]
    by_xi = [{"xi": xi, **summarize_group([r for r in group if r["xi"] == xi], method)}
             for xi, (method, group) in product(plan["xi"], grouped.items())]
    lids = sorted({r["layout_id"] for r in rows})
    bootstrap = np.random.default_rng(plan["streams"]["bootstrap"])
    indices = bootstrap.integers(len(lids), size=(plan["bootstrap_repetitions"], len(lids))) if len(lids) >= 2 else None
    comparisons = [paired_contrast(name, a, b, grouped, lids, indices) for name, a, b in (
        ("validation_selected_vs_baseline", protocol["chosen_group"], "uniform_mse"),
        ("weight_only", "weighted_mse", "uniform_mse"),
        ("physical_selection_only", "uniform_physics", "uniform_mse"),
        ("weight_and_physical_selection", "weighted_physics", "uniform_mse"),
        ("chosen_warm_vs_random", "selected_pso_short", "pso_short"),
        ("chosen_warm_vs_baseline_warm", "selected_pso_short", "baseline_pso_short"),
        ("baseline_warm_vs_random", "baseline_pso_short", "pso_short"))]
    for name, contents in (("comparison_runs", rows), ("comparison_by_method", table),
                            ("comparison_by_xi", by_xi), ("paired_comparison", comparisons)):
        write_table(output / f"{name}.csv", contents)
    return table, by_xi, comparisons


def run_test(output, plan, source, selection):
    freeze = sha256(output / "selection_frozen.json")
    chosen = selection["chosen_group"]
    protocol = {"selection_frozen_sha256": freeze, "chosen_group": chosen, "direct": selection["selected"],
        "warm": {"baseline_pso_short": selection["representatives"]["uniform_mse"],
                 "selected_pso_short": selection["representatives"][chosen]},
        "primary_contrast": "validation-selected group versus uniform-MSE baseline; all three training seeds retained",
        "test_used_for_selection": False}
    write_immutable(output / "test_plan.json", protocol)
    searches = run_phase(output, "test_search", search_layout,
        [(e, plan, source, protocol, output, freeze) for e in plan["test_layouts"]], plan["workers"],
        (output / "test_search_frozen.json").is_file())
    if any(r["selection_frozen_sha256"] != freeze for r in searches.values()):
        raise ValueError("Test search used another checkpoint selection")
    frozen = freeze_phase(output, "test_search", searches, "test_search_frozen.json")
    eligible = [e for e in plan["test_layouts"] if searches[e["layout_id"]]["status"] == "eligible"]
    if not eligible:
        raise ValueError("No new test layout meets the declared eligibility rule")
    print(f"All proposals frozen for {len(eligible)} new test layouts. Starting independent fading audit.", flush=True)
    audits = run_phase(output, "test_audit", audit_layout,
        [(e, plan, source, searches[e["layout_id"]], frozen) for e in eligible], plan["workers"])
    if any(r["test_search_frozen_sha256"] != frozen for r in audits.values()):
        raise ValueError("Test audit belongs to another frozen proposal pool")
    for ref in protocol["direct"].values():
        if sha256(output / ref["path"]) != ref["sha256"]:
            raise ValueError("A selected checkpoint changed during the test")
    rows = [r for lid in sorted(audits) for r in audits[lid]["rows"]]
    marks = [(name, None) for name in protocol["direct"]] + list(product(PSO_METHODS, plan["search_config"]["pso_seeds"]))
    cases = list(product(audits, plan["power_dbm"], plan["xi"]))
    expected = {(*case, *mark) for case, mark in product(cases, marks)}
    actual = {(r["layout_id"], r["total_power_dbm"], r["xi"], r["method"], r["pso_seed"]) for r in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("Test must contain each planned eligible case, method and seed once")
    table, by_xi, comparisons = aggregate(output, plan, protocol, rows)
    return {"step": "6f", "complete": True, "label_slack": plan["label_slack"], "matched_cases": plan["matched_cases"],
        "chosen_group": chosen, "fitted_networks": len(plan["loss_weights"]) * len(plan["training_config"]["seeds"]),
        "validation_candidates": plan["max_validation_candidates"], "test_layouts": len(eligible),
        "test_layouts_planned": len(plan["test_layouts"]), "test_cases": len(cases), "test_audit_rows": len(rows),
        "excluded_test_layouts": {lid: r["status"] for lid, r in searches.items() if r["status"] != "eligible"},
        "unique_test_audit_evaluations": sum(r["unique"] for r in audits.values()),
        "audit_samples_per_layout": plan["audit_samples_per_layout"], "audit_family_size": plan["test_family_size"],
        "test_pso_evaluations": sum(r["evaluations"] for r in rows), "pso_budget": plan["benchmark_budget"],
        "test_used_for_training_or_selection": False,
        "timing_scope": "prebuilt search bank/objective cache; includes graph features, inference, warm particles and PSO; "
                        "excludes offline labels/training/loading/warmup and the independent audit",
        "primary": comparisons[0], "comparisons": comparisons, "methods": table, "by_xi": by_xi}
