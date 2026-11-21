"""Bước 2: Gamma/DEP, ngưỡng Willie, kiểm chứng Rician Monte Carlo.

Chạy từ cùng thư mục với run_step1.py. Không thay đổi tệp bước 1.
"""
from pathlib import Path
import argparse
import csv
import json
import platform
import sys
import time
import numpy as np
import scipy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import gammainc

from ris_noma.config import load_config, dbm_to_w
from ris_noma.geometry import sample_layout
from ris_noma.channels import build_bank, public_order
from ris_noma.random_streams import make_rng, STREAM
from ris_noma.evaluator import evaluate_with_dep, constraint_report
from ris_noma.dep import (gamma_grid, gamma_dep, empirical_dep, empirical_at_threshold,
                          minimize_empirical_dep, gamma_cdf_error, lambertw_threshold_check)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def bank_diagnostics(detail, samples, points, dense_points, tail):
    h0, h1 = detail["h0"], detail["h1"]
    _, _, grid = gamma_grid(h0, h1, points, tail)
    shifts, values, dense = gamma_grid(h0, h1, dense_points, tail)
    optimum = detail["gamma_optimum"]
    if grid["dep_min"] + 1e-10 < optimum["dep_min"] or dense["dep_min"] + 1e-10 < optimum["dep_min"]:
        raise ArithmeticError("Grid tìm được DEP nhỏ hơn solver: kiểm tra nghiệm dừng")
    w = lambertw_threshold_check(h0, h1)
    if w["status"] == "computed":
        actual = optimum["stationary_shifts_w"]
        proposed = w["stationary_shifts_w"]
        w["agrees_with_log_pdf"] = bool(len(actual) == len(proposed) and np.allclose(actual, proposed, rtol=1e-7, atol=0))
        if not w["agrees_with_log_pdf"]:
            w["warning"] = "Lambert W check disagrees; retain log-PDF result and inspect conditioning"
    cdf0 = gammainc(h0.shape, shifts / h0.scale_w)
    cdf1 = gammainc(h1.shape, shifts / h1.scale_w)
    z0, z1 = samples["z0_willie"], samples["z1_willie"]
    report = {"gamma_moments": detail["moments"], "gamma_optimum": optimum,
              "grid": {**grid, "dep_gap_to_stationary_solver": grid["dep_min"] - optimum["dep_min"]},
              "dense_grid": {**dense, "dep_gap_to_stationary_solver": dense["dep_min"] - optimum["dep_min"]},
              "lambert_w_check": w,
              "empirical_minimum_on_this_bank": minimize_empirical_dep(z0, z1),
              "empirical_at_own_gamma_threshold": empirical_at_threshold(optimum["shift_star_w"], z0, z1),
              "cdf_sup_error_h0": gamma_cdf_error(z0, h0), "cdf_sup_error_h1": gamma_cdf_error(z1, h1),
              "gamma_cdf_order_violation_on_dense_grid": float(max(0, np.max(cdf1 - cdf0))),
              "sample_order_Z1_ge_Z0": bool(np.all(z1 >= z0))}
    return report, shifts, values


def make_plots(output, c, details, samples, eval_samples, diagnostics, eval_diagnostics, shifts):
    h0, h1 = details["h0"], details["h1"]
    pref = max(h0.mean_w, h1.mean_w)
    u = shifts / pref
    label = "DEMO parameters" if c["profile"] == "demo" else "Baseline parameters"
    title = f"{label} | K={c['n_public']} | fixed allocation, no PSO"
    optimum = diagnostics["gamma_optimum"]
    mc = diagnostics["empirical_minimum_on_this_bank"]
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    ax.plot(u, gamma_dep(shifts, h0, h1), lw=2, label="Gamma baseline (label moments)")
    ax.plot(u, empirical_dep(shifts, samples["z0_willie"], samples["z1_willie"]), alpha=.85, label="Rician empirical (label)")
    ax.plot(u, empirical_dep(shifts, eval_samples["z0_willie"], eval_samples["z1_willie"]), alpha=.85, label="Rician empirical (evaluation)")
    ax.scatter(optimum["shift_star_w"] / pref, optimum["dep_min"], marker="o", s=45, zorder=4, label="Gamma minimum")
    ax.scatter(mc["shift_w"] / pref, mc["dep"], marker="x", s=60, zorder=4, label="Empirical label minimum")
    ax.set(xlabel="Normalized threshold shift: (lambda - sigma_w^2) / P_ref", ylabel="DEP = P_FA + P_MD", title=title)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output / "dep_vs_threshold.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharex=True, sharey=True)
    for index, ax in enumerate(axes):
        law = [h0, h1][index]
        z = samples[f"z{index}_willie"]
        sorted_z = np.sort(z)
        ax.step(sorted_z / pref, np.arange(1, len(z) + 1) / len(z), where="post", lw=1.5, label="Rician empirical label")
        ax.plot(u, gammainc(law.shape, shifts / law.scale_w), "--", lw=1.8, label="Gamma baseline")
        ax.set(xlabel="Normalized power Z / P_ref", title=f"H{index} | CDF sup error={diagnostics[f'cdf_sup_error_h{index}']:.4f}")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("CDF")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(output / "gamma_vs_rician_cdf.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/demo_k2.json")
    parser.add_argument("--output", default="outputs/step2_k2")
    parser.add_argument("--grid-points", type=int, default=300)
    parser.add_argument("--dense-points", type=int, default=10001)
    parser.add_argument("--tail-tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    start = time.perf_counter()
    c = load_config(args.config)
    layout = sample_layout(c, make_rng(c["root_seed"], c["layout_id"], "geometry"))
    label = build_bank(c, layout, make_rng(c["root_seed"], c["layout_id"], "label"))
    evaluation = build_bank(c, layout, make_rng(c["root_seed"], c["layout_id"], "evaluation"))
    order = public_order(label)
    a = c["reference_allocation"]
    xi = c["requirements"]["xi"][0]
    summary, samples, details = evaluate_with_dep(c, layout, label, order, a["mb"], a["public"], a["s"], xi)
    eligible = summary["bob_strongest_mean_gain_this_bank"]
    eval_summary, eval_samples, eval_details = evaluate_with_dep(c, layout, evaluation, order, a["mb"], a["public"], a["s"], xi, eligible)
    diagnostics, shifts, _ = bank_diagnostics(details, samples, args.grid_points, args.dense_points, args.tail_tolerance)
    eval_diagnostics, _, _ = bank_diagnostics(eval_details, eval_samples, args.grid_points, args.dense_points, args.tail_tolerance)
    fixed_threshold_eval = empirical_at_threshold(details["gamma_optimum"]["shift_star_w"], eval_samples["z0_willie"], eval_samples["z1_willie"])
    fixed_threshold_eval["threshold_selected_on"] = "label_gamma_moments_only"
    fixed_threshold_eval["interpretation"] = "evaluation of a fixed label-selected detector; SE is pointwise, not uncertainty of optimized population DEP"
    rows = []
    for requirement in c["requirements"]["xi"]:
        r0 = constraint_report(summary, requirement, eligible)
        r1 = constraint_report(eval_summary, requirement, eligible)
        empirical_label_pass = bool(diagnostics["empirical_minimum_on_this_bank"]["dep"] >= requirement)
        empirical_eval_pass = bool(eval_diagnostics["empirical_minimum_on_this_bank"]["dep"] >= requirement)
        rows.append({"xi": requirement,
                     "gamma_label_dep_min": summary["dep_min"],
                     "gamma_evaluation_refit_dep_min": eval_summary["dep_min"],
                     "label_constraints": r0, "evaluation_refit_constraints": r1,
                     "empirical_label_minimum_pass_descriptive": empirical_label_pass,
                     "empirical_evaluation_minimum_pass_descriptive": empirical_eval_pass,
                     "label_covertness_diagnostic_disagreement": bool(r0["checks"]["covertness_gamma"] != empirical_label_pass),
                     "evaluation_covertness_diagnostic_disagreement": bool(r1["checks"]["covertness_gamma"] != empirical_eval_pass)})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    sigma_w2 = float(dbm_to_w(c["noise_dbm"]["willie"]))
    metadata = {"step": 2, "profile": c["profile"], "optimized": False, "pso_implemented": False,
                "scope": "fixed allocation evaluator with Gamma DEP and Rician empirical benchmarks",
                "observation_model": "mean power over an asymptotically long time block; random fading; one statistical threshold",
                "variance_model": "baseline_without_covariance_eq_10_5", "sigma_w2_w": sigma_w2,
                "threshold_convention": "H1 iff power > lambda; H0 otherwise; consistent empirical ties",
                "root_seed": c["root_seed"], "layout_id": c["layout_id"], "role_codes": STREAM,
                "public_original_ids_in_sic_order": order.tolist(), "order_fixed_on_evaluation": True,
                "candidate_fixed_on_evaluation": True, "public_power_and_ris_fixed_across_hypotheses": True,
                "public_xy_original_order": layout.public_xy.tolist(),
                "source_equations_v4": ["9.6-9.10", "10.3-10.6", "11.2-11.16"],
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}}
    write_json(output / "config_used.json", c)
    write_json(output / "metadata_step2.json", metadata)
    write_json(output / "metrics_label_step2.json", summary)
    write_json(output / "metrics_evaluation_step2.json", eval_summary)
    write_json(output / "dep_label.json", diagnostics)
    write_json(output / "dep_evaluation.json", eval_diagnostics)
    write_json(output / "fixed_label_threshold_on_evaluation.json", fixed_threshold_eval)
    write_json(output / "feasibility_by_xi.json", rows)
    with (output / "feasibility_by_xi.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["xi", "gamma_label_dep_min", "gamma_eval_refit_dep_min", "label_gamma_sample_feasible", "evaluation_gamma_sample_feasible", "label_empirical_min", "evaluation_empirical_min"])
        for row in rows:
            writer.writerow([row["xi"], row["gamma_label_dep_min"], row["gamma_evaluation_refit_dep_min"], row["label_constraints"]["feasible_under_gamma_sample_model"], row["evaluation_refit_constraints"]["feasible_under_gamma_sample_model"], diagnostics["empirical_minimum_on_this_bank"]["dep"], eval_diagnostics["empirical_minimum_on_this_bank"]["dep"]])
    with (output / "dep_curve.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["shift_w", "lambda_w", "gamma_label_dep", "empirical_label_dep", "empirical_evaluation_dep"])
        for row in zip(shifts, shifts + sigma_w2, gamma_dep(shifts, details["h0"], details["h1"]), empirical_dep(shifts, samples["z0_willie"], samples["z1_willie"]), empirical_dep(shifts, eval_samples["z0_willie"], eval_samples["z1_willie"])):
            writer.writerow(row)
    np.savez_compressed(output / "willie_power_samples.npz", z0_label=samples["z0_willie"], z1_label=samples["z1_willie"], z0_evaluation=eval_samples["z0_willie"], z1_evaluation=eval_samples["z1_willie"])
    make_plots(output, c, details, samples, eval_samples, diagnostics, eval_diagnostics, shifts)
    write_json(output / "timing.json", {"wall_seconds_including_builds_checks_and_figures": time.perf_counter() - start})
    print("PROFILE:", c["profile"].upper(), "| K=", c["n_public"], "| fixed allocation; no PSO")
    print("DEP Gamma minimum [label / eval refit]:", f"{summary['dep_min']:.8f}", "/", f"{eval_summary['dep_min']:.8f}")
    print("lambda* Gamma label [W]:", f"{details['gamma_optimum']['lambda_star_w']:.8e}")
    print("DEP empirical minimum [label / evaluation]:", diagnostics["empirical_minimum_on_this_bank"]["dep"], "/", eval_diagnostics["empirical_minimum_on_this_bank"]["dep"])
    print("DEP at FIXED label Gamma threshold on evaluation:", fixed_threshold_eval["dep"])
    print(f"{args.grid_points}-point grid gap to stationary solver:", f"{diagnostics['grid']['dep_gap_to_stationary_solver']:.3e}")
    print("CDF sup error H0 / H1:", round(diagnostics["cdf_sup_error_h0"], 6), "/", round(diagnostics["cdf_sup_error_h1"], 6))
    print("xi -> feasible under LABEL Gamma/sample model (not certified):")
    for row in rows:
        print(" ", row["xi"], "->", row["label_constraints"]["feasible_under_gamma_sample_model"])
    disagreements = [row["xi"] for row in rows if row["label_covertness_diagnostic_disagreement"] or row["evaluation_covertness_diagnostic_disagreement"]]
    if disagreements:
        print("REVIEW: Gamma and empirical minimum pass decisions differ at xi =", disagreements)
    print("Saved to:", output.resolve())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError) as error:
        print(f"INPUT ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
