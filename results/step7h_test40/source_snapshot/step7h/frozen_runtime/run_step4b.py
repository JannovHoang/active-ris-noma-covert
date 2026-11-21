"""Re-select the existing step-4a proposal pools across xi; fresh heldout audit.

No new PSO runs. Original validation bounds cover all proposals simultaneously.
Source results are read-only. A new heldout seed is required because the old
heldout results have already been examined during development.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import time
import numpy as np

import run_step3b as step3b
import run_step4a as step4a
from ris_noma.config import validate_config
from ris_noma.geometry import sample_layout
from ris_noma.channels import build_bank
from ris_noma.random_streams import make_rng
from ris_noma.validation import fresh_rng, assess_proposal
from ris_noma.cross_xi import verify_validation_records, select_across_xi, target_checks


ROOT = Path(__file__).resolve().parent
PHYSICS = ["ris_noma/channels.py", "ris_noma/geometry.py", "ris_noma/random_streams.py", "ris_noma/metrics.py",
           "ris_noma/dep.py", "ris_noma/evaluator.py", "ris_noma/validation.py", "ris_noma/config.py"]


def source_plan(source, heldout_seed):
    p = step4a.read_json(source / "study_plan.json")
    status = step4a.read_json(source / "study_status.json")
    if p["step"] != "4a" or not status["study_complete"] or status["heldout_used_for_selection"]:
        raise ValueError("Source must be a completed step-4a study with heldout excluded from selection")
    if type(heldout_seed) is not int or heldout_seed < 0 or heldout_seed == p["policy_per_layout"]["heldout_seed"]:
        raise ValueError("Choose a nonnegative heldout seed different from the source study")
    c = validate_config(deepcopy(p["base_config"]))
    if c["n_public"] != 2:
        raise ValueError("Step 4b currently supports K=2")
    for name in PHYSICS:
        if step4a.sha(ROOT / name) != p["code_sha256"][name]:
            raise ValueError(f"Physics/statistical implementation differs from source: {name}")
    L = len(p["study"]["layout_ids"])
    if not np.isclose(p["per_layout_alpha"], p["study"]["study_alpha"] / L):
        raise ValueError("Source study alpha does not reconcile")
    manifest = {name: step4a.sha(source / name) for name in ["study_plan.json", "study_status.json"]}
    for lid in p["study"]["layout_ids"]:
        folder = source / f"layout_{lid:06d}"
        if not step4a.completed_job(folder):
            raise ValueError(f"Incomplete source layout: {lid}")
        child = step4a.read_json(folder / "status.json")
        names = ["completion.json", "status.json", "layout.json"]
        if child["status"] == "completed":
            names += ["plan.json", "config_label_used.json", "proposals_label.json", "validation_all.json", "selected_by_xi.json"]
        for name in names:
            manifest[str(Path(folder.name) / name)] = step4a.sha(folder / name)
    return p, status, manifest


def process_layout(source, output, source_root_plan, lid, heldout_seed):
    src = source / f"layout_{lid:06d}"
    output.mkdir(parents=True, exist_ok=True)
    info = step4a.read_json(src / "layout.json")
    step3b.write_json(output / "layout.json", info)
    old_status = step4a.read_json(src / "status.json")
    if old_status["status"] == "layout_rejected_by_label_bob_mean_gain_rule":
        step3b.write_json(output / "status.json", {"status": "layout_rejected_by_label_bob_mean_gain_rule", "heldout_generated": False})
        return
    c = validate_config(step4a.read_json(src / "config_label_used.json"))
    policy = source_root_plan["policy_per_layout"]
    child_plan = step4a.read_json(src / "plan.json")
    if c["layout_id"] != lid or child_plan["policy"] != policy:
        raise ValueError("Source layout configuration/policy mismatch")
    expected_config = deepcopy(source_root_plan["base_config"])
    expected_config.update(layout_id=lid, n_fading=policy["label_samples"])
    if c != expected_config or not info["bob_mean_gain_rule_on_label"]:
        raise ValueError("Source physical configuration or layout eligibility differs from its study plan")
    order = np.asarray(info["sic_order_original_ids"], int)
    if not np.array_equal(np.sort(order), np.arange(c["n_public"])):
        raise ValueError("Invalid source SIC order")
    proposals = step4a.read_json(src / "proposals_label.json")
    validation = step4a.read_json(src / "validation_all.json")
    alpha = source_root_plan["per_layout_alpha"] / 2
    lookup = verify_validation_records(c, order, proposals, validation, policy["validation_samples"], child_plan["planned_pool_size"], alpha)
    xis = c["requirements"]["xi"]
    selected = select_across_xi(c, order, proposals, validation, xis)
    frozen = {"selection": selected, "selected_using_heldout": False, "selection_source": "all proposals from the same layout, original simultaneous validation bounds",
              "heldout_seed_new": heldout_seed, "alpha_validation_reused": alpha,
              "alpha_heldout_new": alpha, "original_validation_pool_size": len(proposals)}
    step3b.write_json(output / "selection_frozen.json", frozen)
    digest = step4a.sha(output / "selection_frozen.json")
    by_id = {p["proposal_id"]: p for p in proposals}
    ids = {r["proposal_id"] for r in selected if r["proposal_id"] is not None}
    audits = {}
    if ids:
        audit_c = deepcopy(c)
        audit_c["n_fading"] = policy["heldout_samples"]
        layout = sample_layout(c, make_rng(c["root_seed"], lid, "geometry"))
        if not np.allclose(layout.public_xy, info["public_xy_original_order"], rtol=0, atol=1e-12):
            raise ValueError("Regenerated layout differs from source")
        bank = build_bank(audit_c, layout, fresh_rng(c["root_seed"], lid, "heldout", heldout_seed))
        for pid in sorted(ids):
            p = deepcopy(by_id[pid])
            # Assess unique physical allocations once. Numerical bounds are then
            # compared with every assigned xi, rather than reusing a PASS flag.
            p["xi"] = min(r["xi"] for r in selected if r["proposal_id"] == pid)
            audits[pid] = assess_proposal(audit_c, layout, bank, order, p, len(ids), alpha, True)
        del bank
        step3b.write_json(output / "config_heldout_used.json", audit_c)
    step3b.write_json(output / "heldout_selected_unique.json", list(audits.values()))
    old_rows = {r["xi"]: r for r in step4a.read_json(src / "selected_by_xi.json")}
    rows = []
    for selection in selected:
        xi, pid = selection["xi"], selection["proposal_id"]
        old = old_rows[xi]
        row = {"xi": xi, "status": "no_candidate_passed_validation", "proposal_id": pid,
               "source_xi": selection["source_xi"], "source_margin": selection["source_margin"],
               "gamma_target": selection["gamma_target"], "margin": selection["effective_gamma_margin"],
               "feasible_pool_count": selection["feasible_pool_count"],
               "seed": None, "mb": None, "m1": None, "m2": None, "s": None,
               "bob_rate_label": None, "dep_empirical_validation": None, "dep_lower_validation": None,
               "bob_rate_heldout": None, "dep_empirical_heldout": None, "dep_lower_heldout": None,
               "heldout_point_checks_pass": None, "heldout_bound_checks_pass": None, "heldout_used_for_selection": False,
               "previously_selected": old["proposal_id"] is not None,
               "previous_label_rate": old["bob_rate_label"], "paired_label_rate_gain": None,
               "cross_target_reuse": selection["source_xi"] != xi if pid else False}
        if pid is not None:
            p, v, h = by_id[pid], lookup[pid], audits[pid]
            bounded = target_checks(c, order, p, h, xi)
            point = target_checks(c, order, p, h, xi, bounded=False)
            a = p["label_metrics"]["allocation_sic_order"]
            row.update(seed=p["seed"], mb=a["mb"], m1=a["public"][0], m2=a["public"][1], s=a["s"],
                       bob_rate_label=p["label_metrics"]["bob_mean_rate_bit_s_hz"],
                       dep_empirical_validation=v["screen"]["dep_empirical_min"], dep_lower_validation=v["screen"]["dep_uniform_lower"],
                       bob_rate_heldout=h["metrics"]["bob_mean_rate_bit_s_hz"], dep_empirical_heldout=h["screen"]["dep_empirical_min"],
                       dep_lower_heldout=h["screen"]["dep_uniform_lower"], heldout_point_checks_pass=bool(all(point.values())),
                       heldout_bound_checks_pass=bool(all(bounded.values())))
            row["status"] = "selected_on_validation_audit_pass" if row["heldout_bound_checks_pass"] else "selected_on_validation_audit_review"
            if old["proposal_id"] is not None:
                gain = row["bob_rate_label"] - old["bob_rate_label"]
                if gain < -1e-10:
                    raise ArithmeticError("Expanded pool reduced a previously valid best label score")
                row["paired_label_rate_gain"] = gain
            print(f"layout={lid}, xi={xi:.2f}: source_xi={p['xi']:.2f}, source_delta={p['margin']:.3f}, "
                  f"effective_margin={row['margin']:.3f} | heldout lower={row['dep_lower_heldout']:.6f} | {row['status']}", flush=True)
        rows.append(row)
    if step4a.sha(output / "selection_frozen.json") != digest:
        raise ArithmeticError("Selection changed during the new heldout audit")
    step3b.write_json(output / "selected_by_xi.json", rows)
    step3b.write_csv(output / "selected_by_xi.csv", rows)
    step3b.write_json(output / "status.json", {"status": "completed", "selection_sha256_before_and_after_audit": digest,
        "selected_using_heldout": False, "heldout_generated": bool(ids), "new_heldout_seed": heldout_seed,
        "selected_count": len([r for r in rows if r["proposal_id"] is not None]), "unique_heldout_proposals": len(ids),
        "heldout_bound_pass_count": sum(r["heldout_bound_checks_pass"] is True for r in rows),
        "candidate_evaluations": 0, "reused_source_candidate_evaluations": old_status["candidate_evaluations"],
        "validation_reused": True, "validation_pool_size": len(proposals), "alpha_heldout": alpha})


def run(source, output, heldout_seed=2, resume=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output must be separate, non-nested directories")
    old_plan, old_status, manifest = source_plan(source, heldout_seed)
    plan = deepcopy(old_plan)
    plan.update(step="4b", source_study=str(source), source_sha256=manifest, fresh_heldout_seed=heldout_seed,
                reused_source_candidate_evaluations=old_status["completed_layout_candidate_evaluations"],
                new_pso_candidate_evaluations=0,
                new_code_sha256={name: step4a.sha(ROOT / name) for name in ["run_step4b.py", "ris_noma/cross_xi.py", "run_step4a.py"]})
    plan["policy_per_layout"]["heldout_seed"] = heldout_seed
    plan_path = output / "study_plan.json"
    if output.exists() and any(output.iterdir()):
        if not resume or not plan_path.exists() or step4a.read_json(plan_path) != plan:
            raise ValueError("Output is nonempty or source/code/seed changed; use a new output, or --resume with the same plan")
    else:
        output.mkdir(parents=True, exist_ok=True)
        step3b.write_json(plan_path, plan)
    for lid in plan["study"]["layout_ids"]:
        folder = output / f"layout_{lid:06d}"
        if folder.exists() and step4a.completed_job(folder):
            print(f"layout={lid}: verified completed output, skipped", flush=True)
            continue
        if folder.exists() and any(folder.iterdir()):
            step4a.archive_partial(folder)
        started = time.perf_counter()
        try:
            process_layout(source, folder, old_plan, lid, heldout_seed)
            step4a.finalize_job(folder, time.perf_counter() - started)
        except BaseException:
            step4a.summarize(output, plan, draw=False)
            raise
        step4a.summarize(output, plan, draw=False)
    status, summaries = step4a.summarize(output, plan, draw=True)
    cross_rows = []
    for lid in plan["study"]["layout_ids"]:
        path = output / f"layout_{lid:06d}" / "selected_by_xi.json"
        if path.exists():
            cross_rows.extend(dict(layout_id=lid, **r) for r in step4a.read_json(path))
    if cross_rows:
        step3b.write_csv(output / "cross_target_selection.csv", cross_rows)
    status.update(step="4b", new_pso_candidate_evaluations=0,
                  reused_source_candidate_evaluations=old_status["completed_layout_candidate_evaluations"],
                  validation_reused=True, fresh_heldout_seed=heldout_seed,
                  previous_selected_count=sum(r["previously_selected"] for r in cross_rows),
                  recovered_missing_cases=sum(r["proposal_id"] is not None and not r["previously_selected"] for r in cross_rows),
                  cross_target_reuse_count=sum(r["cross_target_reuse"] for r in cross_rows),
                  margin_definition="gamma_target - requested xi; original source_margin is in cross_target_selection.csv")
    step3b.write_json(output / "study_status.json", status)
    return status, summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="../../results/step4a_k2")
    p.add_argument("--output", default="../../results/step4b_k2")
    p.add_argument("--heldout-seed", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    status, _ = run(a.source, a.output, a.heldout_seed, a.resume)
    print("Completed:", status["study_complete"], "| new PSO evaluations: 0 | output:", Path(a.output).resolve())


if __name__ == "__main__":
    main()
