"""PSO K=2 trên bank label cố định; đánh giá nghiệm trên bank độc lập.

Ví dụ: python run_step3.py --config configs/demo_k2.json --label-samples 10000
--eval-samples 50000 --output ../../results/step3_k2 --seeds 1
"""
from copy import deepcopy
from dataclasses import asdict
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

from ris_noma.config import load_config
from ris_noma.geometry import sample_layout
from ris_noma.channels import build_bank, public_order
from ris_noma.random_streams import make_rng, STREAM
from ris_noma.pso import load_pso_settings, optimize
from ris_noma.objective import CachedObjective, decode_k2
from ris_noma.evaluator import evaluate_with_dep
from ris_noma.dep import minimize_empirical_dep, empirical_at_threshold


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows, names=None):
    if names is None:
        names = list(rows[0]) if rows else []
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_selected(c_label, c_eval, layout, label_bank, evaluation_bank, order, record, xi, eligible):
    a = record["label_metrics"]["allocation_sic_order"]
    label, _, details = evaluate_with_dep(c_label, layout, label_bank, order, a["mb"], a["public"], a["s"], xi, eligible)
    fast = record["label_metrics"]
    for name in ["bob_mean_rate_bit_s_hz", "dep_min", "bob_outage_estimate", "public_mean_rate_bit_s_hz"]:
        if not np.allclose(label[name], fast[name], rtol=1e-9, atol=1e-11):
            raise ArithmeticError(f"Cached evaluator và evaluator bước 2 không khớp: {name}")
    evaluation, samples, _ = evaluate_with_dep(c_eval, layout, evaluation_bank, order, a["mb"], a["public"], a["s"], xi, eligible)
    empirical = minimize_empirical_dep(samples["z0_willie"], samples["z1_willie"])
    fixed = empirical_at_threshold(details["gamma_optimum"]["shift_star_w"], samples["z0_willie"], samples["z1_willie"])
    checks_without_dep = [value for key, value in evaluation["constraints"]["checks"].items() if key != "covertness_gamma"]
    empirical_covertness = bool(empirical["dep"] >= xi)
    return {"label_reference_recheck": label, "independent_evaluation": evaluation,
            "evaluation_empirical_minimum": empirical,
            "evaluation_at_fixed_label_gamma_threshold": fixed,
            "evaluation_empirical_covertness_point_estimate_pass": empirical_covertness,
            "evaluation_empirical_full_sample_check_pass": bool(all(checks_without_dep) and empirical_covertness),
            "candidate_adjusted_using_evaluation": False,
            "interpretation": "candidate selected using label only; independent sample estimates are diagnostic, not population certificates"}


def make_plots(output, c, histories, summaries):
    title = f"{c['profile'].upper()} | K=2 | PSO, Gamma baseline constraints"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for xi, seed, history in histories:
        sweeps = [h["sweep"] for h in history]
        rates = [h["gbest_score"] if h["gbest_feasible"] else np.nan for h in history]
        label = f"xi={xi:g}, seed={seed}"
        axes[0].plot(sweeps, rates, label=label, alpha=.8)
        axes[1].plot(sweeps, [h["gbest_violation"] for h in history], label=label, alpha=.8)
    axes[0].set(xlabel="Swarm evaluation sweep (includes initialization)", ylabel="Best feasible Bob rate (bit/s/Hz)")
    axes[1].set(xlabel="Swarm evaluation sweep", ylabel="Best normalized constraint violation")
    for ax in axes:
        ax.grid(alpha=.25)
    if len(histories) <= 12:
        axes[0].legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(output / "pso_convergence.png", dpi=160)
    plt.close(fig)

    good = [row for row in summaries if row["best_seed"] is not None]
    if good:
        good = sorted(good, key=lambda row: row["xi"])
        x = [row["xi"] for row in good]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].plot(x, [r["bob_rate_label"] for r in good], "o-", label="Label Gamma-selected candidate")
        axes[0].plot(x, [r["bob_rate_evaluation"] for r in good], "s--", label="Independent evaluation")
        axes[0].set(xlabel="Required xi", ylabel="Bob mean rate (bit/s/Hz)")
        axes[1].axhline(0, color="black", linestyle=":", label="Required margin >= 0")
        axes[1].plot(x, [r["dep_gamma_label"] - r["xi"] for r in good], "o-", label="Gamma label")
        axes[1].plot(x, [r["dep_gamma_evaluation"] - r["xi"] for r in good], "s--", label="Gamma evaluation")
        axes[1].plot(x, [r["dep_empirical_evaluation_min"] - r["xi"] for r in good], "x-", label="Empirical evaluation minimum")
        axes[1].set(xlabel="Required xi", ylabel="DEP margin (DEP - xi)")
        for ax in axes:
            ax.set_xticks(x)
            ax.grid(alpha=.25)
            ax.legend(fontsize=7)
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(output / "best_found_vs_xi.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/demo_k2.json")
    parser.add_argument("--pso-config", default="configs/pso_pilot.json")
    parser.add_argument("--output", default="outputs/step3_k2")
    parser.add_argument("--label-samples", type=int)
    parser.add_argument("--eval-samples", type=int, default=50000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--xi", nargs="+", type=float)
    args = parser.parse_args()
    started = time.perf_counter()
    c = load_config(args.config)
    if c["n_public"] != 2:
        raise ValueError("Bước 3 hiện triển khai PSO K=2; cần config có n_public=2")
    if args.label_samples is not None:
        if args.label_samples < 2:
            raise ValueError("label-samples cần >=2")
        c["n_fading"] = args.label_samples
    if args.eval_samples < 2 or not args.seeds or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("eval-samples cần >=2; seeds cần không âm và không trùng")
    xis = args.xi if args.xi is not None else c["requirements"]["xi"]
    if len(set(xis)) != len(xis) or any(not np.isfinite(x) or not 0 < x < 1 for x in xis):
        raise ValueError("xi cần khác nhau và thuộc (0,1)")
    if len({int(round(x * 1e6)) for x in xis}) != len(xis):
        raise ValueError("Các mức xi phải phân biệt ở độ phân giải 6 chữ số thập phân của run IDs")
    settings, raw_pso = load_pso_settings(args.pso_config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config_label_used.json", c)
    write_json(output / "pso_config_used.json", raw_pso)
    layout = sample_layout(c, make_rng(c["root_seed"], c["layout_id"], "geometry"))
    label_bank = build_bank(c, layout, make_rng(c["root_seed"], c["layout_id"], "label"))
    order = public_order(label_bank)
    objective = CachedObjective(c, layout, label_bank, order)
    if not objective.layout_eligible:
        write_json(output / "layout_status.json", {"status": "rejected_by_bob_mean_gain_rule", "public_original_ids_in_sic_order": order.tolist()})
        print("LAYOUT REJECTED: Bob mean-gain rule failed; no PSO runs performed.")
        return 0
    c_eval = deepcopy(c)
    c_eval["n_fading"] = args.eval_samples
    evaluation_bank = build_bank(c_eval, layout, make_rng(c["root_seed"], c["layout_id"], "evaluation"))
    write_json(output / "config_evaluation_used.json", c_eval)
    metadata = {
        "step": 3, "profile": c["profile"], "n_public": 2,
        "label_samples": c["n_fading"], "evaluation_samples": c_eval["n_fading"],
        "root_seed": c["root_seed"], "layout_id": c["layout_id"], "run_seeds": args.seeds, "xi_values": xis,
        "rng_pso_entropy_fields": "[root_seed, layout_id, role_pso, run_seed, round(xi*1e6)]",
        "public_original_ids_in_sic_order": order.tolist(), "public_xy_original_order": layout.public_xy.tolist(),
        "pso_settings": asdict(settings), "pso_parameter_status": raw_pso.get("parameter_status", "unspecified"),
        "initial_velocity": f"uniform +/-{settings.initial_velocity_scale} in normalized coordinates" if settings.initial_velocity_scale else "zero",
        "initial_positions": "uniform on normalized box; no injected reference allocation",
        "update": "synchronous; velocity clip then position clip; clipped velocity retained; no constriction",
        "evaluation_count_convention": "swarm_size * sweeps; initialization included; sweeps-1 velocity updates",
        "selection": "feasible first; higher Bob rate if feasible; lower normalized violation otherwise; score breaks violation ties",
        "power_ordering_imposed": False, "dep_variance_model": "baseline_without_covariance_eq_10_5",
        "sample_variance_ddof": c.get("moment_ddof", 1),
        "constraint_tolerance": 0.0, "bank_resampled_per_candidate": False,
        "selected_using_evaluation": False, "candidate_adjusted_using_evaluation": False,
        "interpretation": "best-found numerical reference; no global-optimality or population-feasibility certificate",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}}
    write_json(output / "metadata_step3.json", metadata)
    print(f"PROFILE: {c['profile'].upper()} | K=2 | M_label={c['n_fading']} | M_eval={c_eval['n_fading']}", flush=True)
    print(f"PSO: {settings.swarm_size} particles x {settings.sweeps} sweeps = {settings.swarm_size * settings.sweeps} evaluations/run", flush=True)
    print("PSO parameter status:", raw_pso.get("parameter_status", "unspecified; inspect pso_config_used.json"), flush=True)
    runs, summaries, histories = [], [], []
    for xi in xis:
        tag = f"xi_{xi:.6f}".replace(".", "p")
        per_xi = []
        for seed in args.seeds:
            entropy = [c["root_seed"], c["layout_id"], STREAM["pso"], seed, int(round(xi * 1e6))]
            rng = np.random.default_rng(np.random.SeedSequence(entropy))
            run_start = time.perf_counter()
            result = optimize(lambda z: objective.evaluate(z, xi), 3, settings, rng)
            elapsed = time.perf_counter() - run_start
            record = {"xi": xi, "seed": seed,
                      "status": "feasible_reference_found_on_label" if result.outcome.feasible else "no_feasible_solution_found",
                      "normalized_position": result.position.tolist(), "label_metrics": result.outcome.metrics,
                      "normalized_violation": result.outcome.violation, "candidate_evaluations": result.evaluations,
                      "feasible_candidate_evaluations": result.feasible_evaluations,
                      "optimization_seconds": elapsed, "pso_rng_entropy": entropy}
            per_xi.append(record)
            runs.append(record)
            histories.append((xi, seed, result.history))
            write_json(output / f"{tag}_seed_{seed}_run.json", record)
            write_csv(output / f"{tag}_seed_{seed}_history.csv", result.history)
            write_json(output / "all_runs.json", runs)
            print(f"xi={xi:.2f}, seed={seed}: {record['status']} | Rb={result.outcome.score:.6f} | DEP={result.outcome.metrics['dep_min']:.6f} | evals={result.evaluations} | {elapsed:.2f}s", flush=True)
        feasible = [record for record in per_xi if record["label_metrics"]["feasible_under_gamma_sample_model"]]
        winner = max(feasible, key=lambda record: record["label_metrics"]["bob_mean_rate_bit_s_hz"]) if feasible else None
        stats = {"xi": xi, "run_count": len(per_xi), "feasible_run_count_label": len(feasible),
                 "feasible_run_fraction_label": len(feasible) / len(per_xi),
                 "bob_rate_mean_feasible_runs_label": float(np.mean([r["label_metrics"]["bob_mean_rate_bit_s_hz"] for r in feasible])) if feasible else None,
                 "bob_rate_sd_feasible_runs_label": float(np.std([r["label_metrics"]["bob_mean_rate_bit_s_hz"] for r in feasible], ddof=1)) if len(feasible) > 1 else None}
        write_json(output / f"{tag}_seed_statistics.json", stats)
        row = {"xi": xi, "status": "no_feasible_solution_found", "best_seed": None,
               "mb": None, "m1": None, "m2": None, "s": None,
               "bob_rate_label": None, "public_rate_c1_label": None, "public_rate_c2_label": None,
               "dep_gamma_label": None, "bob_outage_label": None, "label_reference_feasible": None,
               "bob_rate_evaluation": None, "dep_gamma_evaluation": None, "dep_empirical_evaluation_min": None,
               "dep_at_fixed_label_threshold_evaluation": None, "bob_outage_evaluation": None,
               "evaluation_gamma_sample_feasible": None, "evaluation_empirical_full_sample_pass": None,
               "feasible_seed_fraction_label": stats["feasible_run_fraction_label"]}
        if winner is not None:
            checked = evaluate_selected(c, c_eval, layout, label_bank, evaluation_bank, order, winner, xi, objective.layout_eligible)
            write_json(output / f"{tag}_selected_reference_and_evaluation.json", {"selected_seed_using_label_only": winner["seed"], **checked})
            label = checked["label_reference_recheck"]
            evaluation = checked["independent_evaluation"]
            allocation = label["allocation_sic_order"]
            row.update({"status": "selected_using_label_only", "best_seed": winner["seed"],
                        "mb": allocation["mb"], "m1": allocation["public"][0], "m2": allocation["public"][1], "s": allocation["s"],
                        "bob_rate_label": label["bob_mean_rate_bit_s_hz"],
                        "public_rate_c1_label": label["public_mean_rate_bit_s_hz"][0], "public_rate_c2_label": label["public_mean_rate_bit_s_hz"][1],
                        "dep_gamma_label": label["dep_min"], "bob_outage_label": label["bob_outage_estimate"],
                        "label_reference_feasible": label["constraints"]["feasible_under_gamma_sample_model"],
                        "bob_rate_evaluation": evaluation["bob_mean_rate_bit_s_hz"], "dep_gamma_evaluation": evaluation["dep_min"],
                        "dep_empirical_evaluation_min": checked["evaluation_empirical_minimum"]["dep"],
                        "dep_at_fixed_label_threshold_evaluation": checked["evaluation_at_fixed_label_gamma_threshold"]["dep"],
                        "bob_outage_evaluation": evaluation["bob_outage_estimate"],
                        "evaluation_gamma_sample_feasible": evaluation["constraints"]["feasible_under_gamma_sample_model"],
                        "evaluation_empirical_full_sample_pass": checked["evaluation_empirical_full_sample_check_pass"]})
            if not row["label_reference_feasible"]:
                row["status"] = "reference_recheck_infeasible"
            if not row["evaluation_gamma_sample_feasible"] or not row["evaluation_empirical_full_sample_pass"]:
                print(f"  REVIEW xi={xi:.2f}: independent sample checks differ from label feasibility; candidate kept fixed.", flush=True)
            print(f"  Selected seed {winner['seed']} | mb={row['mb']:.6f}, m1={row['m1']:.6f}, m2={row['m2']:.6f}, s={row['s']:.6f} | eval empirical DEP={row['dep_empirical_evaluation_min']:.6f}", flush=True)
        summaries.append(row)
        write_json(output / "best_found_by_xi.json", summaries)
        write_csv(output / "best_found_by_xi.csv", summaries)
    make_plots(output, c, histories, summaries)
    run_rows = [{"xi": r["xi"], "seed": r["seed"], "status": r["status"],
                 "bob_rate_label": r["label_metrics"]["bob_mean_rate_bit_s_hz"], "dep_gamma_label": r["label_metrics"]["dep_min"],
                 "normalized_violation": r["normalized_violation"], "candidate_evaluations": r["candidate_evaluations"],
                 "optimization_seconds": r["optimization_seconds"]} for r in runs]
    write_csv(output / "per_seed_label_results.csv", run_rows)
    write_json(output / "timing_step3.json", {"total_wall_seconds": time.perf_counter() - started,
                                              "optimization_seconds_sum": sum(r["optimization_seconds"] for r in runs),
                                              "pso_candidate_evaluations": sum(r["candidate_evaluations"] for r in runs),
                                              "reference_and_evaluation_checks_included_in_wall_time": True})
    print("Saved to:", output.resolve(), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError) as error:
        print(f"INPUT ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
