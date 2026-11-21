"""Evaluate frozen 6e checkpoints on paired, previously unused test layouts."""
from dataclasses import replace
from itertools import product
import time

import numpy as np

from .benchmark_pso import predict_start, settings_for, warm_positions
from .budget_benchmark import run_phase
from .channels import public_order
from .dep_label_experiment import channel_context, freeze_phase, rng
from .dep_label_learning import read_model, score_cases, write_immutable
from .expanded_benchmark import source_hashes
from .expanded_data import write_table
from .expanded_scenarios import ROOT
from .graph_data import original_allocation, statistical_graph
from .learning_data import sha256
from .learning_train import configure_runtime
from .objective import CachedObjective, decode_k2
from .pso import optimize


PSO_METHODS = ("pso_short", "standard_pso_short", "robust_pso_short")


def search_test_layout(entry, plan, source, protocol, output, training, freeze):
    lid = entry["layout_id"]
    record = {"layout_id": lid, "selection_frozen_sha256": freeze, "status": "invalid_geometry", "rows": []}
    if not entry["geometry_valid"]:
        return record
    base, layout, bank = channel_context(plan, source, entry, "test_search", plan["search_samples_per_layout"])
    order = public_order(bank)
    if not CachedObjective(base, layout, bank, order).layout_eligible:
        return dict(record, status="bob_not_strongest")
    record.update(status="eligible", order=order.tolist())
    c = plan["search_config"]
    configure_runtime(0, c["threads"], "cpu")
    models = {name: read_model(ref, training if name == "gnn_6b" else output) for name, ref in protocol["direct"].items()}
    settings = replace(settings_for(c, "pso_short"), sweeps=plan["benchmark_budget"] // c["swarm_size"])
    for power_idx, power in enumerate(plan["power_dbm"]):
        physical = dict(base, total_dbm=power)
        objective = CachedObjective(physical, layout, bank, order)
        for xi_idx, xi in enumerate(plan["xi"]):
            case = {"layout_id": lid, "total_power_dbm": power, "xi": xi}
            for method, model in models.items():
                started = time.perf_counter()
                graph = statistical_graph(physical, layout, bank, order, xi)
                y, _ = predict_start(*model, graph, order, physical["bounds"])
                record["rows"].append({**case, "method": method, "pso_seed": None, "evaluations": 0,
                                       "allocation": y.tolist(), "search_wall_seconds": time.perf_counter() - started})
            for seed_idx, seed in enumerate(c["pso_seeds"]):
                rotate = (lid + power_idx + xi_idx + seed_idx) % len(PSO_METHODS)
                for method in PSO_METHODS[rotate:] + PSO_METHODS[:rotate]:
                    started = time.perf_counter()
                    initial = None
                    if method in protocol["warm"]:
                        graph = statistical_graph(physical, layout, bank, order, xi)
                        _, z = predict_start(*models[protocol["warm"][method]], graph, order, physical["bounds"])
                        initial = warm_positions(z, c["warm_particles"], c["warm_sigma"],
                                                 rng(plan, base, entry, "warm_jitter", seed, power_idx, xi_idx))
                    target = xi + c["search_dep_margin"]
                    result = optimize(lambda z: objective.evaluate(z, target), 3, settings,
                                      rng(plan, base, entry, "test_pso", seed, power_idx, xi_idx), initial)
                    if result.evaluations != plan["benchmark_budget"]:
                        raise ArithmeticError("Incorrect paired PSO budget")
                    mb, public, s = decode_k2(result.position, physical["bounds"])
                    y, _ = original_allocation({"mb": mb, "public": public, "s": s}, order, physical["bounds"])
                    record["rows"].append({**case, "method": method, "pso_seed": seed, "evaluations": result.evaluations,
                                           "allocation": y.tolist(), "search_wall_seconds": time.perf_counter() - started})
    return record


def audit_test_layout(entry, plan, source, search, family_size, freeze):
    base, layout, bank = channel_context(plan, source, entry, "test_audit", plan["audit_samples_per_layout"])
    rows, unique = score_cases(base, layout, bank, np.asarray(search["order"], int), search["rows"], family_size,
                               plan["test_alpha_per_family"])
    return {"layout_id": entry["layout_id"], "test_search_frozen_sha256": freeze, "rows": rows, "unique": unique}


def mean(rows, field):
    return float(np.mean([row[field] for row in rows]))


def summarize_group(rows, method):
    passed = [row for row in rows if row["audit_pass"]]
    return {"method": method, "runs": len(rows), "passed": len(passed), "pass_fraction": len(passed) / len(rows),
            "gated_bob_rate": mean(rows, "gated_bob_rate"), "bob_rate_passing_mean": mean(passed, "bob_rate") if passed else None,
            "search_wall_ms": 1000. * mean(rows, "search_wall_seconds"),
            "dep_lower_failures": sum("dep_uniform_lower_ge_xi" in row["failed_checks"] for row in rows)}


def aggregate_test(output, plan, protocol, rows):
    seeds = plan["training_config"]["seeds"]
    methods = list(protocol["direct"]) + list(PSO_METHODS)
    grouped = {method: [row for row in rows if row["method"] == method] for method in methods}
    for group in ("standard", "robust"):
        grouped[f"{group}_direct_mean"] = [row for seed in seeds for row in grouped[f"{group}_s{seed}"]]
    report_methods = [*grouped, "standard_direct_selected", "robust_direct_selected"]
    grouped["standard_direct_selected"] = grouped[protocol["representatives"]["standard"]]
    grouped["robust_direct_selected"] = grouped[protocol["representatives"]["robust"]]
    summary = [summarize_group(grouped[method], method) for method in report_methods]
    by_xi = [{"xi": xi, **summarize_group([row for row in grouped[method] if row["xi"] == xi], method)}
             for xi, method in product(plan["xi"], report_methods)]
    lids = sorted({row["layout_id"] for row in rows})
    bootstrap_rng = np.random.default_rng(plan["streams"]["bootstrap"])
    indices = bootstrap_rng.integers(0, len(lids), size=(plan["bootstrap_repetitions"], len(lids))) if len(lids) >= 2 else None
    comparisons = []
    for name, a, b in [
        ("matched_training_seeds", "robust_direct_mean", "standard_direct_mean"),
        ("selected_direct", "robust_direct_selected", "standard_direct_selected"),
        ("robust_warm_vs_random", "robust_pso_short", "pso_short"),
        ("robust_warm_vs_standard", "robust_pso_short", "standard_pso_short"),
    ]:
        a_rows, b_rows = grouped[a], grouped[b]
        result = {"contrast": name, "method_a": a, "method_b": b, "layouts": len(lids), "pairs": len(a_rows)}
        for field, factor in (("audit_pass", 100.), ("gated_bob_rate", 1.), ("search_wall_seconds", 1000.)):
            values = np.array([
                factor * (mean([r for r in a_rows if r["layout_id"] == lid], field)
                          - mean([r for r in b_rows if r["layout_id"] == lid], field))
                for lid in lids
            ])
            metric = {"audit_pass": "pass_delta_pp", "gated_bob_rate": "gated_rate_delta",
                      "search_wall_seconds": "wall_time_delta_ms"}[field]
            result[metric] = float(values.mean())
            low, high = (None, None) if indices is None else np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist()
            result[f"{metric}_ci_low"], result[f"{metric}_ci_high"] = low, high
        comparisons.append(result)
    for name, table in (("comparison_runs", rows), ("comparison_by_method", summary),
                        ("comparison_by_xi", by_xi), ("paired_comparison", comparisons)):
        write_table(output / f"{name}.csv", table)
    return summary, by_xi, comparisons


def run_test(args, output, plan, source, info, refs, selection):
    direct = {"gnn_6b": plan["historical_gnn"]}
    representatives = {}
    for group, slack in (("standard", 0.), ("robust", selection["chosen_slack"])):
        for name, ref in refs.items():
            if ref["slack"] == slack:
                key = f"{group}_s{ref['seed']}"
                direct[key] = ref
                if name == selection["representatives"][group]:
                    representatives[group] = key
    freeze = sha256(output / "selection_frozen.json")
    protocol = {
        "selection_frozen_sha256": freeze, "direct": direct, "representatives": representatives,
        "warm": {f"{group}_pso_short": name for group, name in representatives.items()},
        "primary_contrast": "robust versus standard direct averaged over matched training seeds",
        "code_sha256": sha256(ROOT / "ris_noma/dep_label_benchmark.py"),
    }
    write_immutable(output / "test_plan.json", protocol)
    entries = [e for e in plan["layouts"] if e["split"] == "test"]
    workers = plan["workers"]
    searches = run_phase(output, "test_search", search_test_layout,
                         [(e, plan, source, protocol, output, args.training.resolve(), freeze) for e in entries],
                         workers, (output / "test_search_frozen.json").is_file())
    if any(r["selection_frozen_sha256"] != freeze for r in searches.values()):
        raise ValueError("Test search belongs to a different checkpoint selection")
    search_hash = freeze_phase(output, "test_search", searches, "test_search_frozen.json")
    eligible = [e for e in entries if searches[e["layout_id"]]["status"] == "eligible"]
    if not eligible:
        raise ValueError("No eligible test layouts")
    seeds = plan["search_config"]["pso_seeds"]
    family_size = (len(direct) + len(PSO_METHODS) * len(seeds)) * len(plan["xi"])
    print(f"Test proposals frozen for {len(eligible)} layouts. Starting independent test audit.", flush=True)
    audits = run_phase(output, "test_audit", audit_test_layout,
                       [(e, plan, source, searches[e["layout_id"]], family_size, search_hash) for e in eligible],
                       workers)
    if (any(r["test_search_frozen_sha256"] != search_hash for r in audits.values())
            or source_hashes(args.dataset, args.training) != plan["sources_sha256"]
            or sha256(args.benchmark / "completion.json") != plan["source_6d_sha256"]):
        raise ValueError("Test audit belongs to a different search or source experiment")
    for name, ref in direct.items():
        if sha256((args.training if name == "gnn_6b" else output) / ref["path"]) != ref["sha256"]:
            raise ValueError(f"Frozen test checkpoint changed: {name}")
    rows = [row for lid in sorted(audits) for row in audits[lid]["rows"]]
    marks = [(name, None) for name in direct] + list(product(PSO_METHODS, seeds))
    cases = list(product(audits, plan["power_dbm"], plan["xi"]))
    expected = {(*case, *mark) for case, mark in product(cases, marks)}
    observed = {(r["layout_id"], r["total_power_dbm"], r["xi"], r["method"], r["pso_seed"]) for r in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError("Test audit needs every eligible case, method and seed")
    summary, by_xi, comparisons = aggregate_test(output, plan, protocol, rows)
    return {
        "step": "6e", "complete": True, "chosen_slack": selection["chosen_slack"],
        "label_evaluations": info["label_evaluations"], "matched_labels": info["matched_labels"],
        "test_layouts": len(eligible), "test_layouts_planned": len(entries), "test_cases": len(cases),
        "excluded_test_layouts": {lid: r["status"] for lid, r in searches.items() if r["status"] != "eligible"},
        "test_audit_rows": len(rows), "unique_test_audit_evaluations": sum(r["unique"] for r in audits.values()),
        "test_audit_samples_per_layout": plan["audit_samples_per_layout"], "test_audit_family_size": family_size,
        "test_pso_evaluations": sum(r["evaluations"] for r in rows), "pso_budget": plan["benchmark_budget"],
        "timing_scope": "search bank and objective cache prepared; features, inference, warm particles and PSO included",
        "primary": comparisons[0], "comparisons": comparisons, "methods": summary, "by_xi": by_xi,
    }
