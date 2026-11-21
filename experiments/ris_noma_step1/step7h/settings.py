"""A bounded, preregistered development space and six fixed comparison methods."""
from copy import deepcopy
import numpy as np
from .common import read

DEFAULT = {
    "scope": "main", "short_budgets": [150, 300, 450], "full_budget": 1800,
    "search_samples": 10000, "audit_samples": 100000,
    "search_dep_margin": .03, "search_qos_margin": .01,
    "replicates": [{"training_seed": 7, "pso_seed": 101}, {"training_seed": 17, "pso_seed": 202},
                   {"training_seed": 27, "pso_seed": 303}],
    "cv_seed": 17, "cv_epochs": 200, "low_cv_threshold": .9,
    "alpha": .05, "bootstrap_resamples": 2000, "bootstrap_seed": 7317,
    "budget_max_pass_loss": .01, "budget_min_rate_ratio": .95,
    "fallback": "none", "initialization": "one_prediction_plus_29_random",
    "final_selection": "search_best", "architecture": "small", "training_prefix": 32,
    "smoke_validation_jobs": None, "smoke_cases_per_job": None,
    "smoke_cv_folds": None, "smoke_test_layout_ids": None,
}


def settings(value=None):
    c = deepcopy(DEFAULT if value is None else read(value) if not isinstance(value, dict) else value)
    if set(c) != set(DEFAULT) or c["scope"] not in ("main", "smoke"):
        raise ValueError("Unknown 7h configuration keys/scope")
    for key in ("full_budget", "search_samples", "audit_samples", "cv_epochs", "bootstrap_resamples", "cv_seed", "bootstrap_seed"):
        if type(c[key]) is not int or c[key] < (0 if key in ("cv_seed", "bootstrap_seed") else 1):
            raise ValueError(f"Invalid nonnegative integer setting: {key}")
    budgets = c["short_budgets"]
    if (not budgets or budgets != sorted(set(budgets)) or any(type(v) is not int or v < 30 or v % 30 for v in budgets)
            or c["full_budget"] % 30 or max(budgets) >= c["full_budget"]):
        raise ValueError("Budgets include initialization, must be multiples of 30, and short must be below full")
    reps = c["replicates"]
    if (not reps or any(set(r) != {"training_seed", "pso_seed"} for r in reps)
            or any(type(v) is not int or v < 0 for r in reps for v in r.values())
            or any(r["training_seed"] not in (7, 17, 27) for r in reps)
            or len({r["training_seed"] for r in reps}) != len(reps)
            or len({r["pso_seed"] for r in reps}) != len(reps)):
        raise ValueError("Declare distinct paired training/PSO seeds")
    for key in ("search_dep_margin", "search_qos_margin", "low_cv_threshold", "alpha", "budget_max_pass_loss", "budget_min_rate_ratio"):
        if type(c[key]) not in (int, float) or not np.isfinite(c[key]): raise ValueError("Finite scalar settings required")
    if not 0 < c["alpha"] < 1 or c["search_dep_margin"] < 0 or c["search_qos_margin"] < 0:
        raise ValueError("Invalid margins/alpha")
    if not 0 < c["low_cv_threshold"] < 2 or not 0 <= c["budget_max_pass_loss"] < 1 or not 0 < c["budget_min_rate_ratio"] <= 1:
        raise ValueError("Invalid CV threshold or budget quality targets")
    for name in ("fallback", "initialization", "final_selection", "architecture", "training_prefix"):
        if c[name] != DEFAULT[name]: raise ValueError("The six-method main protocol fixes fallback, initializer, selector and architecture")
    if c["scope"] == "main":
        fixed = ("short_budgets", "full_budget", "search_samples", "audit_samples", "search_dep_margin", "search_qos_margin",
                 "replicates", "cv_seed", "cv_epochs", "low_cv_threshold", "alpha", "budget_max_pass_loss", "budget_min_rate_ratio")
        if any(c[key] != DEFAULT[key] for key in fixed) or any(c[key] is not None for key in c if key.startswith("smoke_")):
            raise ValueError("Main 7h retains the declared development space; use smoke only for reduced verification")
    else:
        if c["search_samples"] < 1000 or c["audit_samples"] < 1000: raise ValueError("Smoke banks require >=1000 samples")
        if (type(c["smoke_validation_jobs"]) is not int or c["smoke_validation_jobs"] < 1
                or type(c["smoke_cases_per_job"]) is not int or c["smoke_cases_per_job"] < 1
                or type(c["smoke_cv_folds"]) is not int or c["smoke_cv_folds"] < 1):
            raise ValueError("Smoke declares its reduced jobs/cases/folds")
        ids = c["smoke_test_layout_ids"]
        if not isinstance(ids, list) or not ids or ids != sorted(set(ids)) or any(type(i) is not int or i < 900000 for i in ids):
            raise ValueError("Smoke test IDs must be explicit IDs >=900000, never the reserved final test")
    return c


def smoke_settings():
    c = deepcopy(DEFAULT)
    c.update(scope="smoke", short_budgets=[30, 60], full_budget=90, search_samples=1000,
             audit_samples=1000, replicates=[{"training_seed": 7, "pso_seed": 101}],
             cv_epochs=2, bootstrap_resamples=200, smoke_validation_jobs=1,
             smoke_cases_per_job=1, smoke_cv_folds=1, smoke_test_layout_ids=[990001])
    return settings(c)


def counts(c, source):
    jobs = [j for j in source["corpus_plan"]["jobs"] if j["split"] == "validation"]
    if c["scope"] == "smoke": jobs = jobs[:c["smoke_validation_jobs"]]
    jobs = deepcopy(jobs)
    for j in jobs:
        if c["scope"] == "smoke": j["cases"] = j["cases"][:c["smoke_cases_per_job"]]
    ids = (source["corpus_plan"]["config"]["reserved_test_layout_ids"] if c["scope"] == "main"
           else c["smoke_test_layout_ids"])
    test_ks = [2, 3, 4, 5, 6] if c["scope"] == "main" else [2, 6]
    folds = [lid for lid, cv in sorted(source["train_cv"].items()) if cv < c["low_cv_threshold"]]
    if c["scope"] == "smoke": folds = folds[:c["smoke_cv_folds"]]
    max_cases = source["corpus_plan"]["config"]["cases_per_layout"] if c["scope"] == "main" else c["smoke_cases_per_job"]
    rep_count = len(c["replicates"])
    views = rep_count * (3 + 3 * len(c["short_budgets"]))
    max_groups = len(jobs) + len(ids) * len(test_ks) + len(folds) * 4
    return {"development_jobs": jobs, "test_layout_ids": ids, "test_public_counts": test_ks,
            "cv_layout_ids": folds, "max_cases_per_job": max_cases,
            "certificate_pool_size": max(4, views) * max_cases,
            "certificate_alpha_per_group": c["alpha"] / (2 * max_groups),
            "certificate_groups_upper_bound": max_groups}
