"""All-case reporting: returned/pass, abstentions, paired layout bootstrap and cost."""
from itertools import product

import numpy as np

from .expanded_data import write_table


def group_summary(rows, method, budget):
    if not rows:
        raise ValueError("Cannot summarize an empty method/budget group")
    passed = [r for r in rows if r["audit_pass"]]
    returned = [r for r in rows if not r["abstained"]]
    def mean(field):
        return float(np.mean([r[field] for r in rows]))
    return {"method": method, "budget": budget, "runs": len(rows), "returned": len(returned),
        "abstained": len(rows) - len(returned), "passed": len(passed),
        "audit_failed_after_return": len(returned) - len(passed), "pass_fraction": len(passed) / len(rows),
        "return_fraction": len(returned) / len(rows),
        "pass_fraction_among_returned": len(passed) / len(returned) if returned else None,
        "gated_bob_rate": mean("gated_bob_rate"),
        "bob_rate_passing_mean": float(np.mean([r["bob_rate"] for r in passed])) if passed else None,
        "online_ms": 1000 * mean("online_seconds"), "standalone_estimate_ms": 1000 * mean("standalone_estimate_seconds"),
        "search_ms": 1000 * mean("search_wall_seconds"), "selection_pool_ms": 1000 * mean("selection_pool_seconds"),
        "search_bank_ms": 1000 * mean("search_bank_seconds"), "selection_bank_ms": 1000 * mean("selection_bank_seconds"),
        "objective_cache_ms": 1000 * mean("objective_cache_seconds"),
        "dep_lower_failures": sum("dep_uniform_lower_ge_xi" in r["failed_checks"] for r in rows)}


def paired_contrast(rows, name, a, b, layout_ids, indices):
    left = [r for r in rows if (r["method"], r["budget"]) == a]
    right = [r for r in rows if (r["method"], r["budget"]) == b]
    result = {"contrast": name, "method_a": a[0], "budget_a": a[1], "method_b": b[0], "budget_b": b[1],
              "layouts": len(layout_ids), "runs_a": len(left), "runs_b": len(right)}
    for field, factor, key in (("audit_pass", 100., "pass_delta_pp"), ("gated_bob_rate", 1., "gated_rate_delta"),
            ("online_seconds", 1000., "online_delta_ms"), ("standalone_estimate_seconds", 1000., "standalone_delta_ms")):
        delta = np.array([factor * (np.mean([r[field] for r in left if r["layout_id"] == lid])
                                    - np.mean([r[field] for r in right if r["layout_id"] == lid])) for lid in layout_ids])
        if not np.isfinite(delta).all():
            raise ValueError("Unpaired layout groups")
        low, high = (None, None) if indices is None else np.quantile(delta[indices].mean(1), [.025, .975]).tolist()
        result[key], result[key + "_ci_low"], result[key + "_ci_high"] = float(delta.mean()), low, high
    return result


def report(output, plan, selection, rows, validation_info, test_info):
    groups = sorted({(r["method"], r["budget"]) for r in rows})
    methods = [group_summary([r for r in rows if (r["method"], r["budget"]) == (method, budget)], method, budget)
               for method, budget in groups]
    by_xi = [{"xi": xi, **group_summary([r for r in rows if r["xi"] == xi and (r["method"], r["budget"]) == (method, budget)], method, budget)}
             for xi, (method, budget) in product(plan["xi"], groups)]
    primary = plan["primary_budget"]
    contrasts = [
        ("screened_warm_vs_screened_cold", ("warm_screened", primary), ("cold_screened", primary)),
        ("selection_effect_warm", ("warm_screened", primary), ("warm_raw", primary)),
        ("selection_effect_cold", ("cold_screened", primary), ("cold_raw", primary)),
        ("raw_warm_vs_raw_cold", ("warm_raw", primary), ("cold_raw", primary)),
        ("screened_warm_vs_direct_gnn", ("warm_screened", primary), ("gnn_raw", 0)),
        ("screened_gnn_vs_direct_gnn", ("gnn_screened", 0), ("gnn_raw", 0)),
        ("screened_warm_vs_screened_full", ("warm_screened", primary), ("full_screened", plan["full_evaluations"]))]
    for a, b in zip(plan["budgets"], plan["budgets"][1:]):
        contrasts.append((f"warm_{a}_vs_cold_{b}", ("warm_screened", a), ("cold_screened", b)))
    lids = sorted(test_info["eligible_layout_ids"])
    random = np.random.default_rng(plan["streams"]["bootstrap"])
    indices = random.integers(len(lids), size=(plan["bootstrap_repetitions"], len(lids))) if len(lids) >= 2 else None
    paired = [paired_contrast(rows, name, a, b, lids, indices) for name, a, b in contrasts]
    headline = [r for r in methods if r["budget"] in (0, primary, plan["full_evaluations"])]
    for name, table in (("comparison_runs", rows), ("comparison_by_method", methods),
                         ("comparison_by_xi", by_xi), ("paired_comparison", paired), ("primary_comparison", headline)):
        write_table(output / f"{name}.csv", table)
    return {"step": "6g", "complete": True, "chosen_search_margin": selection["chosen_search_margin"],
        "test_layouts": len(lids), "test_cases": len(lids) * len(plan["power_dbm"]) * len(plan["xi"]),
        "test_audit_rows": len(rows), "test_pso_evaluations": test_info["pso_evaluations"],
        "selected_model": plan["selected_model"], "primary_budget": primary, "full_evaluations": plan["full_evaluations"],
        "gnn_retrained": False, "test_used_for_selection": False, "abstentions_kept_in_denominators": True,
        "validation_info": validation_info, "test_info": test_info, "primary": paired[0], "comparisons": paired,
        "methods": methods, "primary_methods": headline, "by_xi": by_xi,
        "timing_scope": plan["timing_scope"], "scope": plan["scope"]}
