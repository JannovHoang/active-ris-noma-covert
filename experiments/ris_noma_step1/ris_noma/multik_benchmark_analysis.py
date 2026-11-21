"""Deterministic 7d tables and layout-clustered paired comparisons."""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .expanded_data import write_table
from .learning_data import read_json
from .multik_dataset import read_committed, write_json


def group_rows(rows, names):
    result = defaultdict(list)
    for row in rows: result[tuple(row[name] for name in names)].append(row)
    return result


def aggregate(rows):
    returned = [r for r in rows if r["returned"]]
    margins = [r["dep_lower"] - r["xi"] for r in returned]
    return {"cases": len(rows), "base_layouts": len({r["layout_id"] for r in rows}),
        "passed": sum(r["passed"] for r in rows), "pass_fraction": float(np.mean([r["passed"] for r in rows])),
        "no_return": sum(not r["returned"] for r in rows),
        "gated_rate": float(np.mean([r["gated_bob_rate"] for r in rows])),
        "raw_rate_including_infeasible_returns": float(np.mean([r["bob_rate"] for r in rows])),
        "return_ms_cached_banks": 1000 * float(np.mean([r["return_seconds_cached_banks"] for r in rows])),
        "return_ms_with_bank_generation": 1000 * float(np.mean([r["return_seconds_with_bank_generation"] for r in rows])),
        "mean_objective_evaluations": float(np.mean([r["objective_evaluations"] for r in rows])),
        "dep_lower_margin_median_returned": float(np.median(margins)) if margins else None,
        "dep_lower_margin_min_returned": float(min(margins)) if margins else None}


def table(rows, names):
    return [{**dict(zip(names, key)), **aggregate(part)} for key, part in sorted(group_rows(rows, names).items(), key=lambda p: str(p[0]))]


def paired_bootstrap(rows, repetitions):
    group_names = ("profile", "deployment_id", "search_margin", "selector")
    contrasts = [("gnn_short_pso", "random_short_pso"), ("gnn_short_pso", "mlp_short_pso"),
                 ("gnn_short_pso", "lookup_sic_short_pso"), ("gnn_short_pso", "full_pso")]
    results = []
    for group_index, (group, part) in enumerate(sorted(group_rows(rows, group_names).items(), key=lambda p: str(p[0]))):
        for a, b in contrasts:
            def lookup(method):
                return {(r["layout_id"], r["n_public"], r["power_dbm"], r["xi"], r["training_seed"], r["pso_seed"]): r for r in part if r["method"] == method}
            first, second = lookup(a), lookup(b)
            if first.keys() != second.keys(): raise ValueError("Unpaired method cases")
            clusters = defaultdict(list)
            for key in sorted(first):
                x, y = first[key], second[key]
                clusters[key[0]].append([float(x["passed"]) - float(y["passed"]),
                    x["gated_bob_rate"] - y["gated_bob_rate"],
                    1000 * (x["return_seconds_cached_banks"] - y["return_seconds_cached_banks"])])
            values = np.asarray([np.mean(v, axis=0) for _, v in sorted(clusters.items())])
            rng = np.random.default_rng(np.random.SeedSequence([7301, group_index]))
            boot = values[rng.integers(0, len(values), size=(repetitions, len(values)))].mean(axis=1)
            point = values.mean(axis=0); lower, upper = np.quantile(boot, [.025, .975], axis=0)
            results.append({**dict(zip(group_names, group)), "first": a, "second": b,
                "layout_clusters": len(values), "paired_case_rows": len(first),
                "pass_difference_pp": float(point[0] * 100), "pass_difference_pp_low": float(lower[0] * 100), "pass_difference_pp_high": float(upper[0] * 100),
                "gated_rate_difference": float(point[1]), "gated_rate_difference_low": float(lower[1]), "gated_rate_difference_high": float(upper[1]),
                "cached_latency_difference_ms": float(point[2]), "cached_latency_difference_ms_low": float(lower[2]), "cached_latency_difference_ms_high": float(upper[2]),
                "scope": "exploratory paired percentile bootstrap over base layout IDs; few-cluster pilot, not publication-level evidence"})
    return results


def summarize(output, write=True):
    output = Path(output); plan = read_json(output / "plan.json")
    all_rows = []; probes = []; assumptions = []; search_count = evaluations = 0; los_groups = defaultdict(list)
    for job in plan["jobs"]:
        search = read_committed(output / "search" / (job["id"] + ".json"))
        audit = read_committed(output / "audit" / (job["id"] + ".json"))
        search_count += len(search["pools"]); evaluations += search["objective_evaluations"]
        assumptions.append({"stage": job["stage"], "profile": job["profile"], "deployment_id": job["deployment"]["id"],
            "layout_id": job["layout_id"], "n_public": job["n_public"],
            "bob_strongest_mean_gain_on_search_bank": search["layout_eligible_on_search_bank"]})
        for row in audit["rows"]:
            all_rows.append({**row, "deployment_id": job["deployment"]["id"]})
        for probe in search["fixed_probes"]:
            probes.append({"stage": job["stage"], "profile": job["profile"], "deployment_id": job["deployment"]["id"],
                "layout_id": job["layout_id"], "n_public": job["n_public"], **probe})
        meta = search["metadata"]
        los_groups[(job["stage"], job["profile"], job["deployment"]["id"], job["n_public"])].append((job["layout_id"], meta))
    benchmark = [r for r in all_rows if r["stage"] == "benchmark"]
    calibration = [r for r in all_rows if r["stage"] == "calibration"]
    keys = ("stage", "profile", "deployment_id", "search_margin", "selector", "method")
    overview = table(all_rows, keys)
    by_k = table(all_rows, keys + ("n_public", "regime"))
    by_dep = table(all_rows, keys + ("xi",))
    by_seed = table(benchmark, keys + ("training_seed", "pso_seed"))
    contrasts = paired_bootstrap(benchmark, plan["config"]["bootstrap_repetitions"])
    failure_rows = []
    for key, part in sorted(group_rows(all_rows, keys).items(), key=lambda p: str(p[0])):
        failure_rows.append({**dict(zip(keys, key)), "failed_cases": sum(not r["passed"] for r in part),
            "failed_checks": dict(Counter(name for r in part if not r["passed"] for name in r["failed_checks"]))})
    los_summary = []
    for key, items in sorted(los_groups.items(), key=lambda p: str(p[0])):
        ar = {v["los"]["mode"] + ":" + str(v["los"]["ar_phase_rad"]) for _, v in items}
        bob = {str(v["los"]["ru_phase_rad"][0]) for _, v in items}
        los_summary.append({"stage": key[0], "profile": key[1], "deployment_id": key[2], "n_public": key[3],
            "layouts": len(items), "distinct_ar_vectors": len(ar), "distinct_bob_vectors": len(bob),
            "los_hashes": [v["los_sha256"] for _, v in sorted(items)]})
    result = {"step": "7d", "complete": True, "scope": plan["config"]["scope"],
        "groups": len(plan["jobs"]), "search_pools": search_count, "objective_evaluations": evaluations,
        "audit_rows": len(all_rows), "calibration_rows": len(calibration), "benchmark_rows": len(benchmark),
        "calibration_layouts": len(plan["config"]["calibration_layout_ids"]),
        "benchmark_layouts": len(plan["config"]["benchmark_layout_ids"]),
        "models_per_kind": len(plan["config"]["training_seeds"]), "methods": list(plan["methods"]),
        "selectors": list(plan["selectors"]), "new_weights_trained": False, "final_unseen_test": False,
        "overview": overview, "by_k": by_k, "paired_contrasts": contrasts,
        "limitations": ["frozen models trained on original explicit-LoS regime; geometry results measure transfer, not retrained in-domain performance",
            "geometry_friis is a development link-budget reference, not a measured propagation environment",
            "rates/constraints are conditional on the retained scalar RIS noise and Rician simulation model",
            "few independent layout clusters; no automatic GNN superiority or global optimality claim",
            "timings use shared cached bank computations, with separately reported bank-generation-inclusive figures"]}
    if write:
        for name, rows in (("metrics.csv", overview), ("by_k.csv", by_k), ("by_dep.csv", by_dep),
                           ("by_seed.csv", by_seed), ("paired_contrasts.csv", contrasts), ("link_budget_probes.csv", probes)):
            write_table(output / name, rows)
        write_json(output / "failure_breakdown.json", failure_rows)
        write_json(output / "los_variation.json", los_summary)
        write_table(output / "assumption_checks.csv", assumptions)
        write_json(output / "summary.json", result)
    return result
