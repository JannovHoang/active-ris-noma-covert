"""Summarize paired budget contrasts with layouts as the resampling unit."""
from itertools import product

import numpy as np

from .expanded_data import write_table


METHODS = ("pso_short", "gnn_pso_short")
PAIR_FIELDS = ("layout_id", "total_power_dbm", "xi", "pso_seed")
DELTA_METRICS = {
    "pass_delta_pp": "audit_pass",
    "gated_rate_delta": "gated_bob_rate",
    "wall_time_delta_seconds": "search_wall_seconds",
    "cpu_time_delta_seconds": "search_cpu_seconds",
}


def mean_field(rows, field):
    return float(np.mean([row[field] for row in rows]))


def curve_row(rows, method, budget):
    passed = [row for row in rows if row["audit_pass"]]
    return {
        "method": method, "budget": budget, "runs": len(rows),
        "search_pass_count": sum(row["search_feasible"] for row in rows),
        "audit_pass_count": len(passed), "audit_pass_fraction": len(passed) / len(rows),
        "gated_bob_rate_mean": mean_field(rows, "gated_bob_rate"),
        "bob_rate_passing_mean": mean_field(passed, "bob_rate") if passed else None,
        "search_wall_seconds_mean": mean_field(rows, "search_wall_seconds"),
        "search_cpu_seconds_mean": mean_field(rows, "search_cpu_seconds"),
    }


def paired_layout(warm, cold, lid, gnn_budget, random_budget):
    pairs = list(zip(warm, cold, strict=True))
    row = {
        "layout_id": lid, "gnn_budget": gnn_budget, "random_budget": random_budget, "pairs": len(pairs),
        "gnn_only_pass": sum(a["audit_pass"] and not b["audit_pass"] for a, b in pairs),
        "random_only_pass": sum(b["audit_pass"] and not a["audit_pass"] for a, b in pairs),
        "both_pass": sum(a["audit_pass"] and b["audit_pass"] for a, b in pairs),
    }
    for name, field in DELTA_METRICS.items():
        row[name] = mean_field(warm, field) - mean_field(cold, field)
    row["pass_delta_pp"] *= 100.
    return row


def summarize_pairs(rows, indices, primary):
    head = rows[0]
    result = {
        "gnn_budget": head["gnn_budget"], "random_budget": head["random_budget"],
        "primary": head["gnn_budget"] == head["random_budget"] == primary,
        "layouts": len(rows), "pairs": sum(row["pairs"] for row in rows),
    }
    for field in ("gnn_only_pass", "random_only_pass", "both_pass"):
        result[field] = sum(row[field] for row in rows)
    for field in DELTA_METRICS:
        values = np.array([row[field] for row in rows])
        result[field] = float(values.mean())
        low, high = (None, None) if indices is None else np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist()
        result[f"{field}_ci_low"] = low
        result[f"{field}_ci_high"] = high
    return result


def report(output, plan, rows):
    pairs = {}
    for row in rows:
        key = tuple(row[field] for field in PAIR_FIELDS)
        group = pairs.setdefault(key, {})
        mark = row["method"], row["budget"]
        if mark in group:
            raise ValueError(f"Duplicate budget result for {key}, {mark}")
        group[mark] = row
    lids = sorted({key[0] for key in pairs})
    budgets = plan["budgets"]
    expected_keys = set(product(lids, plan["power_dbm"], plan["xi"], plan["search_config"]["pso_seeds"]))
    expected_marks = set(product(METHODS, budgets))
    if set(pairs) != expected_keys or any(set(group) != expected_marks for group in pairs.values()):
        raise ValueError("Every eligible layout needs all paired cases, methods and budgets")
    keys_by_layout = {lid: sorted(key for key in pairs if key[0] == lid) for lid in lids}
    rng = np.random.default_rng(plan["streams"]["bootstrap"])
    indices = rng.integers(0, len(lids), size=(plan["bootstrap_repetitions"], len(lids))) if len(lids) >= 2 else None
    curve, comparisons, by_layout = [], [], []
    for budget, method in product(budgets, METHODS):
        group = [pair[method, budget] for pair in pairs.values()]
        curve.append(curve_row(group, method, budget))
    primary = plan["primary_budget"]
    contrasts = [(budget, budget) for budget in budgets] + [(primary, plan["cross_budget_reference"])]
    for gnn_budget, random_budget in contrasts:
        group = []
        for lid, keys in keys_by_layout.items():
            warm = [pairs[key]["gnn_pso_short", gnn_budget] for key in keys]
            cold = [pairs[key]["pso_short", random_budget] for key in keys]
            group.append(paired_layout(warm, cold, lid, gnn_budget, random_budget))
        by_layout.extend(group)
        comparisons.append(summarize_pairs(group, indices, primary))
    for name, table in (("comparison_runs", rows), ("curves", curve),
                        ("paired_comparison", comparisons), ("paired_by_layout", by_layout)):
        write_table(output / f"{name}.csv", table)
    return {
        "step": "6d", "complete": True, "planned_layouts": len(plan["layout_ids"]),
        "eligible_layouts": len(lids), "eligible_layout_ids": lids,
        "paired_trajectories": len(pairs), "psovals": len(pairs) * len(METHODS) * budgets[-1],
        "audit_rows": len(rows), "audit_samples_per_layout": plan["audit_samples_per_layout"],
        "primary": next(row for row in comparisons if row["primary"]),
        "cross_budget": comparisons[-1], "curves": curve, "comparisons": comparisons,
        "uncertainty": "95% pointwise percentile bootstrap over paired layout-level means",
    }
