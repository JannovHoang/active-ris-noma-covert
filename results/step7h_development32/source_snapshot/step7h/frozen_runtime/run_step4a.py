"""Multi-layout pilot around step 3b, with fixed study plan and resume support."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import platform
import shutil
import time
import numpy as np
import scipy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_step3b as step3b
from ris_noma.config import load_config
from ris_noma.pso import load_pso_settings


ROOT = Path(__file__).resolve().parent
TRACKED = ["run_step4a.py", "run_step3b.py", "ris_noma/validation.py", "ris_noma/channels.py", "ris_noma/geometry.py",
           "ris_noma/random_streams.py", "ris_noma/metrics.py", "ris_noma/dep.py", "ris_noma/evaluator.py",
           "ris_noma/objective.py", "ris_noma/pso.py", "ris_noma/config.py"]
RECEIPT_FILES = ["status.json", "layout.json", "selection_frozen.json", "selected_by_xi.json", "selected_by_xi.csv"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_study(path):
    study = read_json(path)
    if set(study) - {"layout_ids", "seeds", "study_alpha", "notes"} or {"layout_ids", "seeds", "study_alpha"} - study.keys():
        raise ValueError("Study configuration has missing or unknown keys")
    for name in ["layout_ids", "seeds"]:
        xs = study[name]
        if not isinstance(xs, list) or not xs or len(set(xs)) != len(xs) or any(type(x) is not int or x < 0 for x in xs):
            raise ValueError(f"{name} must be a nonempty list of distinct nonnegative integers")
    if not np.isfinite(study["study_alpha"]) or not 0 < study["study_alpha"] < 1:
        raise ValueError("study_alpha must lie in (0,1)")
    return study


def make_plan(c, settings, policy, study):
    n = len(study["layout_ids"])
    p = deepcopy(policy)
    p["seeds"] = list(study["seeds"])
    p["alpha"] = study["study_alpha"] / n
    specs, skipped = step3b.make_specs(c["requirements"]["xi"], p)
    return {"step": "4a", "base_config": c, "pso_settings": asdict(settings), "policy_per_layout": p,
            "study": study, "per_layout_alpha": p["alpha"], "planned_candidate_evaluations_per_eligible_layout": len(specs) * settings.swarm_size * settings.sweeps,
            "planned_candidate_evaluations_if_all_layouts_eligible": n * len(specs) * settings.swarm_size * settings.sweeps,
            "skipped_targets": skipped, "code_sha256": {name: sha(ROOT / name) for name in TRACKED},
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
            "scope": "predeclared finite study under the simulator; conditional layout eligibility is checked on label; exploratory study, not the final GNN test set"}


def completed_job(folder):
    receipt = folder / "completion.json"
    if not receipt.exists():
        return False
    record = read_json(receipt)
    for name, digest in record["sha256"].items():
        if not (folder / name).is_file() or sha(folder / name) != digest:
            raise ValueError(f"Completed result changed: {folder / name}. Use a new study output or restore the recorded file.")
    return True


def finalize_job(folder, elapsed):
    status = read_json(folder / "status.json")
    if status["status"] not in ["completed", "layout_rejected_by_label_bob_mean_gain_rule"]:
        raise ValueError("Child run did not complete or reject its layout explicitly")
    if status["status"] == "completed":
        if sha(folder / "selection_frozen.json") != status["selection_sha256_before_and_after_audit"]:
            raise ValueError("Selection hash does not match child status")
        if status["selected_using_heldout"]:
            raise ValueError("Heldout must not select or repair candidates")
    names = [name for name in RECEIPT_FILES if (folder / name).exists()]
    temporary = folder / "completion.json.tmp"
    step3b.write_json(temporary, {"job_wall_seconds": elapsed, "sha256": {name: sha(folder / name) for name in names}})
    temporary.replace(folder / "completion.json")


def archive_partial(folder):
    i = 1
    while folder.with_name(f"{folder.name}_partial_{i:03d}").exists():
        i += 1
    destination = folder.with_name(f"{folder.name}_partial_{i:03d}")
    shutil.move(str(folder), str(destination))
    return destination


def summarize(output, plan, draw=True):
    """Keep all planned layouts visible; rate means include every selected audit."""
    records, layouts = [], []
    ids = plan["study"]["layout_ids"]
    xis = plan["base_config"]["requirements"]["xi"]
    elapsed, evaluations = 0., 0
    for layout_id in ids:
        folder = output / f"layout_{layout_id:06d}"
        done = completed_job(folder) if folder.exists() else False
        info = read_json(folder / "layout.json") if done else None
        state = read_json(folder / "status.json") if done else None
        rejected = bool(state and state["status"] == "layout_rejected_by_label_bob_mean_gain_rule")
        layout_state = "pending_or_incomplete" if not done else "layout_rejected" if rejected else "completed"
        pos = info["public_xy_original_order"] if info else [[None, None], [None, None]]
        layout_row = {"layout_id": layout_id, "state": layout_state, "alpha": plan["per_layout_alpha"],
                      "public0_x_m": pos[0][0], "public0_y_m": pos[0][1], "public1_x_m": pos[1][0], "public1_y_m": pos[1][1],
                      "sic_rank1_original_id": info["sic_order_original_ids"][0] if info else None,
                      "sic_rank2_original_id": info["sic_order_original_ids"][1] if info else None}
        layouts.append(layout_row)
        if done:
            elapsed += read_json(folder / "completion.json")["job_wall_seconds"]
            evaluations += state.get("candidate_evaluations", 0)
        children = read_json(folder / "selected_by_xi.json") if done and not rejected else []
        lookup = {r["xi"]: r for r in children}
        if done and not rejected and set(lookup) != set(xis):
            raise ValueError("Completed child rows do not match planned xi values")
        for xi in xis:
            child = lookup.get(xi)
            record = {"layout_id": layout_id, "xi": xi, "layout_state": layout_state,
                      "result_status": child["status"] if child else layout_state,
                      "proposal_id": child["proposal_id"] if child else None,
                      "seed": child["seed"] if child else None, "margin": child["margin"] if child else None,
                      "mb": child["mb"] if child else None, "m1": child["m1"] if child else None,
                      "m2": child["m2"] if child else None, "s": child["s"] if child else None,
                      "bob_rate_heldout": child["bob_rate_heldout"] if child else None,
                      "dep_empirical_heldout": child["dep_empirical_heldout"] if child else None,
                      "dep_lower_heldout": child["dep_lower_heldout"] if child else None,
                      "heldout_point_checks_pass": child["heldout_point_checks_pass"] if child else None,
                      "heldout_bound_checks_pass": child["heldout_bound_checks_pass"] if child else None,
                      "heldout_used_for_selection": child["heldout_used_for_selection"] if child else False}
            records.append(record)
    summaries = []
    for xi in xis:
        relevant = [r for r in records if r["xi"] == xi]
        eligible = [r for r in relevant if r["layout_state"] == "completed"]
        selected = [r for r in eligible if r["proposal_id"] is not None]
        passes = sum(r["heldout_bound_checks_pass"] is True for r in selected)
        rates = [r["bob_rate_heldout"] for r in selected]  # Include audit failures.
        summaries.append({"xi": xi, "planned_layouts": len(ids), "eligible_completed_layouts": len(eligible),
                          "rejected_layouts": sum(r["layout_state"] == "layout_rejected" for r in relevant),
                          "pending_or_incomplete_layouts": sum(r["layout_state"] == "pending_or_incomplete" for r in relevant),
                          "validation_selected_count": len(selected), "no_candidate_passed_validation_count": len(eligible) - len(selected),
                          "heldout_point_pass_count": sum(r["heldout_point_checks_pass"] is True for r in selected),
                          "heldout_bound_pass_count": passes, "heldout_review_count": len(selected) - passes,
                          "bound_pass_fraction_all_requested_layouts": passes / len(ids),
                          "bound_pass_fraction_selected": passes / len(selected) if selected else None,
                          "bob_rate_heldout_mean_all_selected": float(np.mean(rates)) if rates else None,
                          "bob_rate_heldout_sd_all_selected": float(np.std(rates, ddof=1)) if len(rates) > 1 else None,
                          "minimum_heldout_dep_lower_margin_selected": min(r["dep_lower_heldout"] - xi for r in selected) if selected else None,
                          "mean_selected_margin": float(np.mean([r["margin"] for r in selected])) if selected else None})
    status = {"study_complete": all(r["state"] != "pending_or_incomplete" for r in layouts),
              "planned_layouts": len(ids), "finalized_layouts": sum(r["state"] != "pending_or_incomplete" for r in layouts),
              "label_rule_rejected_layouts": sum(r["state"] == "layout_rejected" for r in layouts),
              "completed_layout_candidate_evaluations": evaluations, "completed_layout_wall_seconds_sum": elapsed,
              "study_alpha": plan["study"]["study_alpha"], "per_layout_alpha": plan["per_layout_alpha"],
              "heldout_used_for_selection": False,
              "rate_aggregation": "all validation-selected candidates, including heldout audit failures; no post-hoc filtering of rate means"}
    step3b.write_csv(output / "all_layout_results.csv", records)
    step3b.write_csv(output / "layout_positions.csv", layouts)
    step3b.write_csv(output / "summary_by_xi.csv", summaries)
    step3b.write_json(output / "study_status.json", status)
    if draw and any(s["validation_selected_count"] for s in summaries):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
        available = [s for s in summaries if s["validation_selected_count"]]
        x = [s["xi"] for s in available]
        axes[0].errorbar(x, [s["bob_rate_heldout_mean_all_selected"] for s in available],
                         yerr=[s["bob_rate_heldout_sd_all_selected"] or 0 for s in available], fmt="o-", capsize=4)
        axes[0].set(xlabel="Required xi", ylabel="Heldout Bob rate: mean +/- sample SD (bit/s/Hz)")
        axes[1].plot([s["xi"] for s in summaries], [s["validation_selected_count"] for s in summaries], "o-", label="Selected on validation")
        axes[1].plot([s["xi"] for s in summaries], [s["heldout_bound_pass_count"] for s in summaries], "s--", label="Passed heldout bounds")
        axes[1].set(xlabel="Required xi", ylabel="Layout count", ylim=(-.2, len(ids) + .5))
        axes[1].legend(fontsize=8)
        for ax in axes:
            ax.set_xticks(xis)
            ax.grid(alpha=.25)
        fig.suptitle(f"Exploratory study: {status['finalized_layouts']}/{len(ids)} layouts finalized", fontsize=11)
        fig.tight_layout()
        fig.savefig(output / "study_summary.png", dpi=170)
        plt.close(fig)
    return status, summaries


def run_study(c, settings, policy, study, output, resume=False, run_layout=None):
    output = Path(output)
    plan = make_plan(c, settings, policy, study)
    plan_path = output / "study_plan.json"
    if output.exists() and any(output.iterdir()):
        if not resume or not plan_path.exists():
            raise ValueError("Nonempty study output: use --resume with the same plan, or choose a new output")
        if read_json(plan_path) != plan:
            raise ValueError("Study plan/configuration/code changed; use a new output directory")
    else:
        output.mkdir(parents=True, exist_ok=True)
        step3b.write_json(plan_path, plan)
    run_layout = run_layout or step3b.run
    for layout_id in study["layout_ids"]:
        folder = output / f"layout_{layout_id:06d}"
        if folder.exists() and completed_job(folder):
            print(f"layout_id={layout_id}: already completed, verified and skipped", flush=True)
            continue
        if folder.exists() and any(folder.iterdir()):
            archived = archive_partial(folder)
            print(f"layout_id={layout_id}: partial output retained at {archived.name}; restarting this layout", flush=True)
        current = deepcopy(c)
        current["layout_id"] = layout_id
        start = time.perf_counter()
        print(f"START layout_id={layout_id}, PSO seeds={study['seeds']}, alpha={plan['per_layout_alpha']:.6g}", flush=True)
        try:
            run_layout(current, settings, deepcopy(plan["policy_per_layout"]), c["requirements"]["xi"], folder)
            finalize_job(folder, time.perf_counter() - start)
        except BaseException:
            summarize(output, plan, draw=False)
            raise
        summarize(output, plan, draw=False)
    status, summaries = summarize(output, plan, draw=True)
    for r in summaries:
        print(f"xi={r['xi']:.2f}: selected={r['validation_selected_count']}, heldout bound pass={r['heldout_bound_pass_count']}, "
              f"eligible completed={r['eligible_completed_layouts']}, rejected={r['rejected_layouts']}", flush=True)
    print("Study results:", output.resolve(), flush=True)
    return status, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/legacy_physics_k2_pilot.json")
    parser.add_argument("--pso-config", default="configs/pso_inherited_coefficients.json")
    parser.add_argument("--validation-config", default="configs/validation_pilot.json")
    parser.add_argument("--study-config", default="configs/study_pilot.json")
    parser.add_argument("--output", default="outputs/step4a")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    c = load_config(args.config)
    settings, _ = load_pso_settings(args.pso_config)
    policy = step3b.load_policy(args.validation_config)
    study = load_study(args.study_config)
    run_study(c, settings, policy, study, args.output, args.resume)


if __name__ == "__main__":
    main()
