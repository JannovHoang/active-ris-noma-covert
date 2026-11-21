"""Evaluator bước 2: nối metric bước 1 với DEP Gamma baseline."""
import numpy as np
from .config import dbm_to_w
from .metrics import evaluate_candidate
from .dep import fit_willie, minimize_gamma_dep


def constraint_report(summary, xi, layout_eligible=None):
    if not np.isfinite(xi) or not 0 < xi < 1:
        raise ValueError("xi phải trong (0,1)")
    if layout_eligible is None:
        layout_eligible = summary["bob_strongest_mean_gain_this_bank"]
    checks = {
        "covertness_gamma": bool(summary["dep_min"] >= xi),
        "public_average_qos": bool(all(summary["qos_pass_per_public"])),
        "bob_outage_point_estimate": bool(summary["bob_outage_point_estimate_pass"]),
        "positive_sic_ceiling_margins": bool(summary["outage_ceiling_margins_positive"]),
        "layout_bob_mean_gain_rule": bool(layout_eligible),
    }
    return {"xi": float(xi), "checks": checks,
            "feasible_under_gamma_sample_model": bool(all(checks.values())),
            "dep_margin": float(summary["dep_min"] - xi),
            "numeric_constraint_tolerance": 0.0,
            "interpretation": "sample estimates plus baseline Gamma approximation; not certified population feasibility"}


def evaluate_with_dep(c, layout, bank, order, mb, mp, s, xi, layout_eligible=None):
    summary, samples = evaluate_candidate(c, layout, bank, order, mb, mp, s)
    h0, h1, z0, z1, moments = fit_willie(
        samples["xw"], samples["yw"], summary["power"]["pa_w"], mb, c["kappa"],
        layout.loss_ar, float(layout.loss_ru[-1]), float(dbm_to_w(c["noise_dbm"]["ris"])),
        ddof=c.get("moment_ddof", 1))
    optimum = minimize_gamma_dep(h0, h1)
    sigma_w2 = float(dbm_to_w(c["noise_dbm"]["willie"]))
    optimum["lambda_star_w"] = sigma_w2 + optimum["shift_star_w"]
    summary["dep_min"] = optimum["dep_min"]
    summary["lambda_star_w"] = optimum["lambda_star_w"]
    summary["dep_model"] = moments["variance_model"]
    summary["constraints"] = constraint_report(summary, xi, layout_eligible)
    summary["full_feasibility"] = "sample_approx_feasible" if summary["constraints"]["feasible_under_gamma_sample_model"] else "sample_approx_infeasible"
    samples = {**samples, "z0_willie": z0, "z1_willie": z1}
    return summary, samples, {"h0": h0, "h1": h1, "moments": moments, "gamma_optimum": optimum}
