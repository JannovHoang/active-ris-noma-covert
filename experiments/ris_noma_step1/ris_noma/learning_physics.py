"""Reproduce source audits and forward-evaluate frozen network allocations.

Keep the existing Active RIS/SIC/DEP evaluator unchanged. Physics evaluation
never changes a model, its reference labels, or its predicted allocation.
"""
from copy import deepcopy
from pathlib import Path
import csv
import json
import time
import numpy as np
from .learning_data import (read_json, write_json, sha256, file_fingerprints,
                            load_split, physics_allocation, TARGET_KEY)
from .config import validate_config
from .geometry import sample_layout
from .channels import build_bank, public_order
from .random_streams import make_rng
from .graph_data import statistical_graph, original_allocation
from .objective import CachedObjective
from .evaluator import evaluate_with_dep
from .validation import fresh_rng, statistical_screen


def records_for(root):
    return [json.loads(line) for line in (Path(root) / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def regenerate_label_context(root, layout_id, split):
    """Reconcile geometry, SIC, graph inputs and label metrics before any audit."""
    manifest = read_json(Path(root) / "dataset_manifest.json")
    c = validate_config(deepcopy(manifest["base_physics_config"]))
    c.update(layout_id=int(layout_id), n_fading=int(manifest["label_bank_samples_per_layout"]))
    layout = sample_layout(c, make_rng(c["root_seed"], c["layout_id"], "geometry"))
    bank = build_bank(c, layout, make_rng(c["root_seed"], c["layout_id"], "label"))
    order = public_order(bank)
    objective = CachedObjective(c, layout, bank, order)
    if not objective.layout_eligible:
        raise ValueError("Saved layout does not reproduce the label eligibility rule")
    data = load_split(root, split)
    indices = np.flatnonzero(data["layout_id"] == layout_id)
    graph = statistical_graph(c, layout, bank, order, float(data["xi"][indices[0]]))
    lookup = {(r["layout_id"], r["xi"]): r for r in records_for(root) if r["label_available"]}
    for index in indices:
        if not np.array_equal(data["sic_order_original_ids"][index], order):
            raise ValueError("SIC order cannot be reproduced")
        expected_global = graph["global_raw"].copy(); expected_global[0] = data["xi"][index]
        if not np.allclose(graph["node_raw"], data["node_raw"][index], rtol=1e-9, atol=1e-10) or not np.allclose(expected_global, data["global_raw"][index], rtol=1e-9, atol=1e-10):
            raise ValueError("Regenerated label features differ from the exported inputs")
        r = lookup[(int(layout_id), float(data["xi"][index]))]
        result = objective.evaluate(data["y_pso_sic"][index], r["source_xi"] + r["source_margin"])
        for key in ["bob_mean_rate_bit_s_hz", "dep_min", "bob_outage_estimate"]:
            if not np.isclose(result.metrics[key], r["label_metrics"][key], rtol=1e-9, atol=1e-10):
                raise ValueError(f"Regenerated label metric differs: {key}")
    del objective, bank
    return c, layout, order


def assess_allocation(c, layout, bank, order, allocation_original, xi, pool_size, alpha):
    y = physics_allocation(allocation_original, c["bounds"])
    public_sic = y[1:3][order]
    summary, samples, _ = evaluate_with_dep(c, layout, bank, order, y[0], public_sic, y[3], float(xi), True)
    screen = statistical_screen(c, summary, samples, order, float(xi), int(pool_size), float(alpha))
    return summary, screen


def reproduce_audits(dataset, output):
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise ValueError("Audit output already exists; use a new file")
    before = file_fingerprints(dataset)
    manifest = read_json(dataset / "dataset_manifest.json")
    records = records_for(dataset)
    reviews = [r for r in records if r.get("heldout_bound_checks_pass") is False]
    policy = manifest["label_generation_policy"]
    n_layouts = sum(len(v) for v in manifest["partition"]["splits"].values())
    alpha = manifest["source_study_alpha"] / n_layouts / 2
    rows = []
    for r in reviews:
        c, layout, order = regenerate_label_context(dataset, r["layout_id"], r["split"])
        c["n_fading"] = policy["heldout_samples"]
        unique = {x["proposal_id"] for x in records if x["layout_id"] == r["layout_id"] and x["label_available"]}
        bank = build_bank(c, layout, fresh_rng(c["root_seed"], c["layout_id"], "heldout", policy["heldout_seed"]))
        summary, screen = assess_allocation(c, layout, bank, order, r["allocation_original_order"], r["xi"], len(unique), alpha)
        values = {"dep_empirical_heldout": screen["dep_empirical_min"], "dep_lower_heldout": screen["dep_uniform_lower"],
                  "bob_rate_heldout": summary["bob_mean_rate_bit_s_hz"]}
        matches = {k: bool(np.isclose(v, r["audit_summary"][k], rtol=1e-9, atol=1e-10)) for k, v in values.items()}
        if not all(matches.values()) or screen["accepted"] != r["heldout_bound_checks_pass"]:
            raise ValueError("Reproduced audit differs from its recorded values")
        rows.append({"layout_id": r["layout_id"], "xi": r["xi"], "split": r["split"],
                     "proposal_id": r["proposal_id"], "heldout_seed": policy["heldout_seed"],
                     "n_fading": c["n_fading"], "pool_size": len(unique), "alpha": alpha,
                     "recorded_values_reproduced": matches, "failed_checks": [k for k, ok in screen["checks"].items() if not ok],
                     "metrics": summary, "screen": screen, "label_retained": True})
        print(f"Audit layout={r['layout_id']} xi={r['xi']}: reproduced; failed={rows[-1]['failed_checks']}", flush=True)
        del bank
    result = {"source_audit_reproduced": True, "reviews": rows, "source_dataset_unchanged": file_fingerprints(dataset) == before,
              "scope": "reproduction of the original source audit; not a new independent audit"}
    output.parent.mkdir(parents=True, exist_ok=True); write_json(output, result)
    return result


def evaluate_physics(dataset, predictions, output, samples=100000, bank_seed=3, alpha=.05,
                     splits=("validation", "test")):
    dataset, predictions, output = Path(dataset), Path(predictions), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Physics output is nonempty; use a new output directory")
    if type(samples) is not int or samples < 2 or type(bank_seed) is not int or bank_seed < 0 or not 0 < alpha < 1:
        raise ValueError("Invalid evaluation samples, seed or alpha")
    if not splits or len(set(splits)) != len(splits) or any(s not in ["train", "validation", "test"] for s in splits):
        raise ValueError("Unknown or repeated physical evaluation splits")
    manifest = read_json(dataset / "dataset_manifest.json")
    previous_seeds = {manifest["label_generation_policy"]["heldout_seed"], 1}
    if bank_seed in previous_seeds:
        raise ValueError("Use a fresh heldout bank seed, different from source/audit seeds 1 and 2")
    evaluation_plan = read_json(predictions / "evaluation_plan.json")
    before = file_fingerprints(dataset)
    if before != evaluation_plan["dataset_sha256"]:
        raise ValueError("Predictions belong to a different dataset")
    prediction_files = sorted([p for s in splits for p in predictions.glob(f"predictions_*_{s}.npz")])
    if not prediction_files:
        raise ValueError("No frozen prediction files")
    loaded = {s: {} for s in splits}
    for role in splits:
        data = load_split(dataset, role)
        for p in prediction_files:
            suffix = "_" + role + ".npz"
            if not p.name.endswith(suffix):
                continue
            name = p.name[len("predictions_"):-len(suffix)]
            with np.load(p, allow_pickle=False) as a:
                if not np.array_equal(a["layout_id"], data["layout_id"]) or not np.array_equal(a["xi"], data["xi"]) or not np.allclose(a["reference"], data[TARGET_KEY], rtol=0, atol=0):
                    raise ValueError("Prediction key/reference mismatch")
                if a["prediction"].shape != data[TARGET_KEY].shape:
                    raise ValueError("Prediction shape mismatch")
                loaded[role][name] = a["prediction"].copy()
        if not loaded[role]:
            raise ValueError(f"Missing predictions for {role}")
    layout_ids = sorted({int(i) for role in splits for i in load_split(dataset, role)["layout_id"]})
    output.mkdir(parents=True, exist_ok=True)
    plan = {"dataset_sha256": before, "prediction_sha256": {p.name: sha256(p) for p in prediction_files},
            "samples_per_layout": samples, "bank_seed": bank_seed, "alpha_total": alpha,
            "splits": list(splits), "layout_ids": layout_ids, "checkpoint_selection_on_this_bank": False,
            "family": "per-layout planned allocations from every declared model/seed/xi plus PSO references",
            "scope": "new fading audit on existing pilot layouts; not an independent new-layout final GNN test"}
    write_json(output / "physics_plan.json", plan)
    rows, start = [], time.perf_counter()
    for role in splits:
        data = load_split(dataset, role)
        variants = {**loaded[role], "pso_reference": data[TARGET_KEY]}
        for lid in sorted(set(data["layout_id"].tolist())):
            c, layout, order = regenerate_label_context(dataset, lid, role)
            c["n_fading"] = samples
            bank = build_bank(c, layout, fresh_rng(c["root_seed"], lid, "heldout", bank_seed))
            indices = np.flatnonzero(data["layout_id"] == lid)
            family_size = len(variants) * len(indices)
            for name, allocation in variants.items():
                for k in indices:
                    summary, screen = assess_allocation(c, layout, bank, order, allocation[k], data["xi"][k], family_size, alpha / len(layout_ids))
                    mp_original = np.empty(2); mp_original[order] = summary["public_mean_rate_bit_s_hz"]
                    y = physics_allocation(allocation[k], c["bounds"])
                    _, z = original_allocation({"mb": y[0], "public": y[1:3][order].tolist(), "s": y[3]}, order, c["bounds"])
                    rows.append({"model": name, "split": role, "layout_id": lid, "xi": float(data["xi"][k]),
                                 "allocation_original_order": y.tolist(), "warm_start_pso_sic": z.tolist(),
                                 "bob_rate": summary["bob_mean_rate_bit_s_hz"], "public_rates_original_order": mp_original.tolist(),
                                 "dep_gamma": summary["dep_min"], "dep_empirical": screen["dep_empirical_min"],
                                 "dep_lower": screen["dep_uniform_lower"], "dep_lower_margin": screen["dep_lower_margin"],
                                 "bob_outage": summary["bob_outage_estimate"], "bob_outage_upper": screen["bob_outage_upper"],
                                 "public_rate_lower_sic_order": screen["public_rate_lower_sic_order"],
                                 "point_gamma_feasible": summary["constraints"]["feasible_under_gamma_sample_model"],
                                 "point_empirical_feasible": bool(summary["constraints"]["feasible_under_gamma_sample_model"] and screen["dep_empirical_min"] >= data["xi"][k]),
                                 "bound_checks_pass": screen["accepted"], "checks": screen["checks"],
                                 "failed_checks": [key for key, ok in screen["checks"].items() if not ok]})
            print(f"Physics {role} layout={lid}: {len(variants)} variants x {len(indices)} xi, M={samples}", flush=True)
            del bank
    summary_rows = []
    for role in splits:
        for name in sorted({r["model"] for r in rows if r["split"] == role}):
            group = [r for r in rows if r["split"] == role and r["model"] == name]
            passed = [r for r in group if r["bound_checks_pass"]]
            summary_rows.append({"model": name, "split": role, "n": len(group),
                                 "point_gamma_feasible": sum(r["point_gamma_feasible"] for r in group),
                                 "point_empirical_feasible": sum(r["point_empirical_feasible"] for r in group),
                                 "bound_pass_count": len(passed), "bound_pass_fraction": len(passed) / len(group),
                                 "bob_rate_mean_all_predictions": float(np.mean([r["bob_rate"] for r in group])),
                                 "bob_rate_mean_bound_passing_only": float(np.mean([r["bob_rate"] for r in passed])) if passed else None,
                                 "min_dep_lower_margin": min(r["dep_lower_margin"] for r in group),
                                 "max_bob_outage_upper": max(r["bob_outage_upper"] for r in group)})
    with (output / "physics_results.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
    with (output / "physics_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0])); w.writeheader(); w.writerows(summary_rows)
    result = {"physics_evaluation_complete": True, "rows": len(rows), "summary": summary_rows,
              "elapsed_seconds": time.perf_counter() - start,
              "source_dataset_unchanged": file_fingerprints(dataset) == before,
              "predictions_unchanged": all(sha256(predictions / n) == digest for n, digest in plan["prediction_sha256"].items()),
              "warm_start_status": "PSO coordinates exported; no short-PSO experiment performed at step 5b",
              "final_gnn_test": False}
    write_json(output / "physics_summary.json", result)
    return result
