"""K=2 PSO proposals on label, margin selection on validation, fresh final audit.

Run from the experiment directory. This is a pilot for ONE fixed layout, not
the bulk GNN dataset generator. All proposals are frozen before validation.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import time
import numpy as np
import scipy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ris_noma.config import load_config
from ris_noma.channels import build_bank, public_order
from ris_noma.geometry import sample_layout
from ris_noma.random_streams import make_rng, STREAM
from ris_noma.objective import CachedObjective
from ris_noma.pso import load_pso_settings, optimize
from ris_noma.validation import fresh_rng, assess_proposal, select_on_validation, family_constants


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def load_policy(path):
    p = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    required = {"label_samples", "validation_samples", "heldout_samples", "margins", "seeds", "alpha", "validation_seed", "heldout_seed"}
    if required - p.keys() or p.keys() - required - {"notes"}:
        raise ValueError("Validation policy has missing or unknown keys")
    for name in ["label_samples", "validation_samples", "heldout_samples"]:
        if type(p[name]) is not int or p[name] < 2:
            raise ValueError(f"{name} must be an integer >= 2")
    for name in ["validation_seed", "heldout_seed"]:
        if type(p[name]) is not int or p[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not p["seeds"] or len(set(p["seeds"])) != len(p["seeds"]) or any(type(x) is not int or x < 0 for x in p["seeds"]):
        raise ValueError("seeds must be distinct nonnegative integers")
    margins = p["margins"]
    if not margins or any(not np.isfinite(x) or x < 0 for x in margins) or len({round(x * 1e6) for x in margins}) != len(margins):
        raise ValueError("margins must be nonnegative and distinct at 1e-6 resolution")
    family_constants(2, 1, 2, p["alpha"])
    return p


def make_specs(xis, policy):
    if not xis or len(set(xis)) != len(xis) or any(not np.isfinite(x) or not 0 < x < 1 for x in xis):
        raise ValueError("xi values must be distinct and lie in (0,1)")
    if len({round(x * 1e6) for x in xis}) != len(xis):
        raise ValueError("xi values must be distinct at 1e-6 resolution")
    specs, skipped = [], []
    for xi in xis:
        for margin in policy["margins"]:
            target = float(xi + margin)
            if target >= 1:
                skipped.append({"xi": xi, "margin": margin, "reason": "xi_plus_margin_ge_1"})
                continue
            for seed in policy["seeds"]:
                specs.append({"proposal_id": f"p{len(specs):04d}", "xi": float(xi), "margin": float(margin),
                              "gamma_target": target, "seed": seed})
    if not specs:
        raise ValueError("No valid xi+margin target remains")
    return specs, skipped


def baseline_rows(path):
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    result = {}
    for r in rows:
        key = float(r["xi"])
        if key in result:
            raise ValueError("Baseline CSV contains duplicate xi values")
        result[key] = r
    return result


def plot_results(output, rows):
    good = [r for r in rows if r["proposal_id"] is not None]
    if not good:
        return
    x = [r["xi"] for r in good]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    axes[0].axhline(0, color="black", ls=":")
    for key, label, style in [("dep_gamma_label", "Gamma label", "o-"),
                               ("dep_empirical_validation", "Empirical validation", "s-"),
                               ("dep_lower_validation", "Lower bound validation", "s--"),
                               ("dep_empirical_heldout", "Empirical heldout", "x-"),
                               ("dep_lower_heldout", "Lower bound heldout", "x--")]:
        axes[0].plot(x, [r[key] - r["xi"] for r in good], style, label=label)
    axes[0].set(xlabel="Required xi", ylabel="DEP minus required xi")
    axes[1].plot(x, [r["bob_rate_heldout"] for r in good], "o-", label="Step 3b: fresh heldout")
    with_baseline = [r for r in good if r["baseline_bob_rate_label"] is not None]
    if with_baseline:
        axes[1].plot([r["xi"] for r in with_baseline], [r["baseline_bob_rate_label"] for r in with_baseline], "s--", label="Earlier step 3: label reference")
    axes[1].set(xlabel="Required xi", ylabel="Bob mean rate (bit/s/Hz)")
    for ax in axes:
        ax.set_xticks(x)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle("One-layout pilot: validation selection, fresh heldout audit", fontsize=11)
    fig.tight_layout()
    fig.savefig(output / "validation_and_rate.png", dpi=170)
    plt.close(fig)


def run(c, settings, policy, xis, output, baseline=None):
    """Separate phases. Heldout is created only after selection_frozen.json."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    c = deepcopy(c)
    if c["n_public"] != 2:
        raise ValueError("Step 3b currently supports K=2")
    c["n_fading"] = policy["label_samples"]
    specs, skipped = make_specs(xis, policy)
    pool_size = len(specs)  # All planned candidates, including failed label runs.
    stage_alpha = policy["alpha"] / 2
    plan = {"step": "3b", "n_public": 2, "policy": policy, "proposal_specs": specs, "skipped_targets": skipped,
            "planned_pool_size": pool_size, "pso_settings": asdict(settings),
            "candidate_evaluations_planned": pool_size * settings.swarm_size * settings.sweeps,
            "overall_alpha_budget_one_invocation": policy["alpha"], "alpha_per_stage": stage_alpha,
            "validation_expected_dep_subtraction": family_constants(policy["validation_samples"], pool_size, 2, stage_alpha)["dep_subtraction"],
            "validation_rng_entropy": [c["root_seed"], c["layout_id"], 6, policy["validation_seed"]],
            "heldout_rng_entropy": [c["root_seed"], c["layout_id"], 7, policy["heldout_seed"]],
            "old_step3_evaluation_stream_reused": False,
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}}
    write_json(output / "plan.json", plan)
    write_json(output / "config_label_used.json", c)
    layout = sample_layout(c, make_rng(c["root_seed"], c["layout_id"], "geometry"))
    label_bank = build_bank(c, layout, make_rng(c["root_seed"], c["layout_id"], "label"))
    order = public_order(label_bank)
    objective = CachedObjective(c, layout, label_bank, order)
    write_json(output / "layout.json", {"public_xy_original_order": layout.public_xy.tolist(), "sic_order_original_ids": order.tolist(),
                                          "bob_mean_gain_rule_on_label": objective.layout_eligible})
    if not objective.layout_eligible:
        write_json(output / "status.json", {"status": "layout_rejected_by_label_bob_mean_gain_rule", "heldout_generated": False})
        print("LAYOUT REJECTED by label Bob mean-gain rule", flush=True)
        return []
    print(f"PSO label only: {pool_size} proposals x {settings.swarm_size * settings.sweeps} evaluations", flush=True)
    proposals = []
    for spec in specs:
        entropy = [c["root_seed"], c["layout_id"], STREAM["pso"], spec["seed"], int(round(spec["gamma_target"] * 1e6))]
        result = optimize(lambda z: objective.evaluate(z, spec["gamma_target"]), 3, settings,
                          np.random.default_rng(np.random.SeedSequence(entropy)))
        proposal = {**spec, "label_feasible": bool(result.outcome.feasible), "label_metrics": result.outcome.metrics,
                    "normalized_position": result.position.tolist(), "pso_rng_entropy": entropy,
                    "candidate_evaluations": result.evaluations, "violation": result.outcome.violation}
        proposals.append(proposal)
        write_csv(output / f"{spec['proposal_id']}_history.csv", result.history)
        write_json(output / "proposals_label.json", proposals)
        print(f"  {spec['proposal_id']}: xi={spec['xi']:.2f}, delta={spec['margin']:.3f}, seed={spec['seed']}, label_pass={result.outcome.feasible}", flush=True)
    # Cached objective owns sliced gain arrays; release the large raw label bank.
    del label_bank, objective
    cv = deepcopy(c)
    cv["n_fading"] = policy["validation_samples"]
    write_json(output / "config_validation_used.json", cv)
    validation_bank = build_bank(cv, layout, fresh_rng(c["root_seed"], c["layout_id"], "validation", policy["validation_seed"]))
    validations = [assess_proposal(cv, layout, validation_bank, order, p, pool_size, stage_alpha, True) for p in proposals]
    write_json(output / "validation_all.json", validations)
    write_csv(output / "margin_candidates.csv", [
        {"proposal_id": p["proposal_id"], "xi": p["xi"], "margin": p["margin"], "seed": p["seed"],
         "gamma_target": p["gamma_target"], "label_feasible": p["label_feasible"],
         "bob_rate_label": p["label_metrics"]["bob_mean_rate_bit_s_hz"], "dep_gamma_label": p["label_metrics"]["dep_min"],
         "dep_gamma_validation": v["metrics"]["dep_min"], "dep_empirical_validation": v["screen"]["dep_empirical_min"],
         "dep_lower_validation": v["screen"]["dep_uniform_lower"], "validation_screen_pass": v["screen"]["accepted"],
         "failed_validation_checks": "; ".join(k for k, ok in v["screen"]["checks"].items() if not ok)}
        for p, v in zip(proposals, validations)])
    del validation_bank
    selections = select_on_validation(proposals, validations, xis)
    frozen = {"selection": selections, "ranking": "highest label Bob rate among candidates passing validation; ties use smaller margin then seed/id",
              "selected_using_heldout": False, "alpha_validation": stage_alpha, "validation_pool_size": pool_size}
    write_json(output / "selection_frozen.json", frozen)
    selection_hash = hashlib.sha256((output / "selection_frozen.json").read_bytes()).hexdigest()
    winners = [p for p in proposals if p["proposal_id"] in {s["proposal_id"] for s in selections}]
    audit_lookup = {}
    if winners:
        ch = deepcopy(c)
        ch["n_fading"] = policy["heldout_samples"]
        write_json(output / "config_heldout_used.json", ch)
        heldout_bank = build_bank(ch, layout, fresh_rng(c["root_seed"], c["layout_id"], "heldout", policy["heldout_seed"]))
        audits = [assess_proposal(ch, layout, heldout_bank, order, p, len(winners), stage_alpha, True) for p in winners]
        write_json(output / "heldout_selected_only.json", audits)
        audit_lookup = {a["proposal_id"]: a for a in audits}
        del heldout_bank
    validation_lookup = {v["proposal_id"]: v for v in validations}
    proposal_lookup = {p["proposal_id"]: p for p in proposals}
    rows = []
    for selection in selections:
        xi, pid = selection["xi"], selection["proposal_id"]
        old = (baseline or {}).get(xi)
        row = {"xi": xi, "status": selection["status"], "proposal_id": pid, "seed": None, "margin": None,
               "gamma_target": None, "mb": None, "m1": None, "m2": None, "s": None,
               "bob_rate_label": None, "dep_gamma_label": None,
               "dep_empirical_validation": None, "dep_lower_validation": None, "validation_screen_pass": False,
               "bob_rate_heldout": None, "dep_gamma_heldout": None, "dep_empirical_heldout": None, "dep_lower_heldout": None,
               "heldout_point_checks_pass": None, "heldout_bound_checks_pass": None,
               "bob_outage_heldout": None, "bob_outage_upper_heldout": None,
               "baseline_bob_rate_label": float(old["bob_rate_label"]) if old else None,
               "label_rate_reduction_percent_vs_baseline": None, "heldout_used_for_selection": False}
        if pid is not None:
            p, v, h = proposal_lookup[pid], validation_lookup[pid], audit_lookup[pid]
            a = p["label_metrics"]["allocation_sic_order"]
            point = bool(h["metrics"]["constraints"]["feasible_under_gamma_sample_model"] and h["screen"]["dep_empirical_min"] >= xi)
            row.update(seed=p["seed"], margin=p["margin"], gamma_target=p["gamma_target"], mb=a["mb"], m1=a["public"][0], m2=a["public"][1], s=a["s"],
                       bob_rate_label=p["label_metrics"]["bob_mean_rate_bit_s_hz"], dep_gamma_label=p["label_metrics"]["dep_min"],
                       dep_empirical_validation=v["screen"]["dep_empirical_min"], dep_lower_validation=v["screen"]["dep_uniform_lower"], validation_screen_pass=True,
                       bob_rate_heldout=h["metrics"]["bob_mean_rate_bit_s_hz"], dep_gamma_heldout=h["metrics"]["dep_min"],
                       dep_empirical_heldout=h["screen"]["dep_empirical_min"], dep_lower_heldout=h["screen"]["dep_uniform_lower"],
                       heldout_point_checks_pass=point, heldout_bound_checks_pass=h["screen"]["accepted"],
                       bob_outage_heldout=h["metrics"]["bob_outage_estimate"], bob_outage_upper_heldout=h["screen"]["bob_outage_upper"])
            row["status"] = "selected_on_validation_audit_pass" if h["screen"]["accepted"] else "selected_on_validation_audit_review"
            if old and float(old["bob_rate_label"]) != 0:
                row["label_rate_reduction_percent_vs_baseline"] = 100 * (1 - row["bob_rate_label"] / float(old["bob_rate_label"]))
            print(f"xi={xi:.2f}: delta={p['margin']:.3f} | validation lower DEP={v['screen']['dep_uniform_lower']:.6f} | "
                  f"fresh heldout empirical/lower={h['screen']['dep_empirical_min']:.6f}/{h['screen']['dep_uniform_lower']:.6f} | {row['status']}", flush=True)
        else:
            print(f"xi={xi:.2f}: no candidate passed validation; this is not proof of infeasibility", flush=True)
        rows.append(row)
    write_csv(output / "selected_by_xi.csv", rows)
    write_json(output / "selected_by_xi.json", rows)
    plot_results(output, rows)
    if hashlib.sha256((output / "selection_frozen.json").read_bytes()).hexdigest() != selection_hash:
        raise ArithmeticError("Selection changed during heldout evaluation")
    write_json(output / "status.json", {"status": "completed", "selection_sha256_before_and_after_audit": selection_hash,
               "heldout_generated": bool(winners), "selected_using_heldout": False,
               "selected_count": len(winners), "heldout_bound_pass_count": sum(r["heldout_bound_checks_pass"] is True for r in rows),
               "candidate_evaluations": sum(p["candidate_evaluations"] for p in proposals), "elapsed_seconds": time.perf_counter() - start,
               "scope": "one fixed layout under the simulator. Confidence budgets apply to this finite family in this invocation, not future layouts or repeated tuning."})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/legacy_physics_k2_pilot.json")
    parser.add_argument("--pso-config", default="configs/pso_inherited_coefficients.json")
    parser.add_argument("--validation-config", default="configs/validation_pilot.json")
    parser.add_argument("--baseline-csv")
    parser.add_argument("--output", default="outputs/step3b")
    args = parser.parse_args()
    c = load_config(args.config)
    settings, _ = load_pso_settings(args.pso_config)
    policy = load_policy(args.validation_config)
    old = baseline_rows(args.baseline_csv)
    run(c, settings, policy, c["requirements"]["xi"], args.output, old)
    print("Saved to:", Path(args.output).resolve(), flush=True)


if __name__ == "__main__":
    main()
