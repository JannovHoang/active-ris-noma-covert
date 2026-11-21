"""Development -> immutable protocol -> new final test, with explicit role barriers."""
from copy import deepcopy
from pathlib import Path
from shutil import copy2
import time
from filelock import FileLock

from . import FORMAT
from .common import (METHODS, SHORT, read, write, digest, freeze, verify, code_hashes, versions, separate, now,
                     committed, safe_path, snapshot_code, ROOT)
from .settings import settings, smoke_settings, counts
from .source import bind, snapshot, source_diagnostic
from .benchmark import run_stage, all_rows
from .analysis import summarize, budget_selection
from . import cvcheck
from ris_noma.continuous_data import parameter_cases
from ris_noma.learning_train import configure_runtime


def inspect(corpus, training, diagnostics, config=None):
    configure_runtime(0, 1, "cpu")
    s = bind(corpus, training, diagnostics)
    c = settings(config)
    workload = counts(c, s)
    cases = sum(len(j["cases"]) for j in workload["development_jobs"])
    test_cases = len(workload["test_layout_ids"])*len(workload["test_public_counts"])*workload["max_cases_per_job"]
    return {"step": "7h", "scope": c["scope"], "source_7g_32_valid": True,
        "methods": list(METHODS), "reused_main_checkpoints": 6, "architecture": "small",
        "checkpoint_epochs_and_deltas": {k:{n:r[n] for n in ("epoch", "delta", "parameters")} for k,r in s["models"].items()},
        "training_layouts": 32, "labeled_training_cases": sum(r["has_label"] for r in s["train"]),
        "development_layouts": len({j["layout_id"] for j in workload["development_jobs"]}),
        "development_scenarios": cases,
        "development_objective_evaluations": cases*len(c["replicates"])*(c["full_budget"]+3*sum(c["short_budgets"])),
        "new_diagnostic_fits": 2*len(workload["cv_layout_ids"]), "cv_layout_ids": workload["cv_layout_ids"],
        "reserved_test_layouts": workload["test_layout_ids"], "reserved_test_scenarios": test_cases,
        "test_case_count_is_planned_not_generated": True, "final_test_generated": False,
        "new_labels_required": False, "new_main_model_training_required": False,
        "search_samples": c["search_samples"], "audit_samples": c["audit_samples"],
        "short_budgets": c["short_budgets"], "full_budget": c["full_budget"],
        "certificate_pool_size": workload["certificate_pool_size"],
        "certificate_alpha_per_group": workload["certificate_alpha_per_group"]}


def initial_plan(source, c, workers, stage):
    if type(workers) is not int or not 1 <= workers <= 4: raise ValueError("Use 1..4 CPU workers; 2 recommended")
    workload = counts(c, source)
    jobs = workload["development_jobs"]
    if stage == "timing":
        # Geometry is already part of validation; use the largest observed K and one case.
        lid = min(j["layout_id"] for j in jobs)
        job = deepcopy(max((j for j in jobs if j["layout_id"] == lid), key=lambda j:j["n_public"]))
        job["cases"] = job["cases"][:1]
        jobs = [job]
    return {"format": FORMAT, "stage": stage, "settings": c, "workers": workers,
        "jobs": jobs, "short_budgets": c["short_budgets"], "pso_settings": source["corpus_plan"]["pso_settings"],
        "source_hashes": source["source_hashes"], "source_directories": {k:str(source[k]) for k in ("corpus", "training", "diagnostics")},
        "code_sha256": code_hashes(), "versions": versions(),
        **{key:workload[key] for key in ("test_layout_ids", "test_public_counts", "cv_layout_ids", "max_cases_per_job",
             "certificate_pool_size", "certificate_alpha_per_group", "certificate_groups_upper_bound")},
        "main_model_policy": "Original 7g/32 small physical checkpoint per seed; delta from completed 7g diagnostics; no new main model fitting",
        "selection_policy": "Same search-best rule for all PSO methods; all outcomes committed before independent audit; no fallback",
        "test_used_for_selection": False, "reserved_test_generated": False,
        "plan_amendment": "Choose the completed 32-layout prefix for final comparison; original 7g learning-curve seal requiring >=2 sizes is not claimed completed"}


def prepare(corpus, training, diagnostics, output, config, workers, resume, stage):
    configure_runtime(0, 1, "cpu")
    s = bind(corpus, training, diagnostics)
    c = settings(config)
    output = Path(output).resolve()
    separate(output, s["corpus"], s["training"], s["diagnostics"])
    separate(output, ROOT / "step7h")
    plan = initial_plan(s, c, workers, stage)
    if output.exists() and any(output.iterdir()):
        if not (output / "plan.json").exists() or read(output / "plan.json") != plan:
            raise ValueError("Nonempty output belongs to another plan; use a new output")
        if not resume and not (output / "completion.json").exists():
            raise ValueError("An incomplete run exists; add --resume to continue the same plan")
    output.mkdir(parents=True, exist_ok=True)
    write(output / "plan.json", plan, immutable=True)
    snapshot_code(output)
    snapshot(output, s)
    source_diagnostic(output, s)
    return output, s, plan


def completion(output, plan):
    output = Path(output)
    paths = [p for p in sorted(output.rglob("*")) if p.is_file() and p.name != "protocol.json"
             and p != output / "completion.json" and "figures" not in p.relative_to(output).parts
             and not p.name.endswith((".tmp", ".lock"))]
    freeze(output, "completion.json", paths, plan_sha256=digest(output / "plan.json"),
           stage=plan["stage"], final_test_generated=plan["stage"] == "final")


def develop(corpus, training, diagnostics, output, config=None, workers=2, resume=False):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output)+".lock", timeout=0):
        output, source, plan = prepare(corpus, training, diagnostics, output, config, workers, resume, "development")
        if (output / "completion.json").exists():
            from .checks import check
            return dict(check(output), already_complete=True)
        started = time.perf_counter()
        run_stage(output, plan)
        summary = summarize(output, plan, all_rows(output, plan))
        print(f"7h validation budget selected: {summary['budget_selection']['selected_short_budget']}; quality target met={summary['budget_selection']['reference_target_met']}", flush=True)
        print("7h exploratory low-CV folds; main checkpoints stay frozen", flush=True)
        diagnostic = cvcheck.run(output, plan)
        # Check source manifests/inputs again before completing; never modify those directories.
        after = bind(corpus, training, diagnostics)
        if after["source_hashes"] != source["source_hashes"]: raise ValueError("Source changed during development")
        write(output / "run_timing.json", {"this_invocation_seconds": time.perf_counter()-started,
            "note": "One invocation wall time; after resume this is not the sum over all invocations"})
        completion(output, plan)
        from .checks import check
        return check(output)


def pilot(corpus, training, diagnostics, output, config=None, workers=2, resume=False):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output)+".lock", timeout=0):
        output, source, plan = prepare(corpus, training, diagnostics, output, config, workers, resume, "timing")
        if (output / "completion.json").exists():
            verify(output, read(output / "completion.json"))
            return read(output / "timing_pilot.json")
        started = time.perf_counter()
        run_stage(output, plan)
        job = plan["jobs"][0]
        search = committed(output / f"search/{job['id']}.json")
        audit = committed(output / f"audit/{job['id']}.json")
        calls = search["objective_evaluations"]
        per_call = sum(r["pso_seconds"] for r in search["rows"])/calls if calls else None
        per_audit = audit["scoring_seconds"]/audit["unique_allocation_evaluations"] if audit["unique_allocation_evaluations"] else None
        work = counts(plan["settings"], source)
        dev_cases = sum(len(j["cases"]) for j in work["development_jobs"])
        test_cases = len(work["test_layout_ids"])*len(work["test_public_counts"])*work["max_cases_per_job"]
        reps = len(plan["settings"]["replicates"])
        bank_seconds = search["metadata"].get("bank_generation_seconds", 0.)+audit["metadata"].get("bank_generation_seconds", 0.)
        estimates = None
        if per_call is not None and per_audit is not None:
            dev_calls = dev_cases*reps*(plan["settings"]["full_budget"]+3*sum(plan["short_budgets"]))
            dev_rows = dev_cases*reps*(3+3*len(plan["short_budgets"]))
            dev_sec = (dev_calls*per_call+dev_rows*per_audit+len(work["development_jobs"])*bank_seconds)/workers
            final = [(test_cases*reps*(plan["settings"]["full_budget"]+3*b)*per_call
                      +test_cases*reps*6*per_audit+len(work["test_layout_ids"])*len(work["test_public_counts"])*bank_seconds)/workers/60
                     for b in (min(plan["short_budgets"]), max(plan["short_budgets"]))]
            estimates = {"development_search_and_audit_minutes": dev_sec/60,
                         "final_minutes_for_min_and_max_short_budget": final,
                         "cv_training_not_included": True}
        result = {"scope": plan["settings"]["scope"], "timing_job": job["id"], "this_invocation_seconds": time.perf_counter()-started,
            "measured_objective_seconds_per_call": per_call, "measured_physics_seconds_per_unique_allocation": per_audit,
            "input_bank_seconds": search["metadata"].get("bank_generation_seconds"),
            "audit_bank_seconds": audit["metadata"].get("bank_generation_seconds"),
            "linear_estimates": estimates, "workers": workers, "final_test_generated": False,
            "note": "Timing only; not used for checkpoint/budget selection. Linear estimates may change with K, contention, caching and laptop temperature; not an ETA"}
        write(output / "timing_pilot.json", result)
        completion(output, plan)
        return result


def seal(output):
    from .checks import check
    output = Path(output).resolve()
    with FileLock(str(output)+".lock", timeout=0):
        verified = check(output)
        if not verified["complete"] or verified["stage"] != "development":
            raise ValueError("Finish the development benchmark and CV diagnostic before sealing")
        plan = read(output / "plan.json")
        rows = all_rows(output, plan)
        choice = budget_selection(rows, plan["settings"])
        if choice != read(output / "budget_selection.json"): raise ValueError("Budget was not selected by the fixed validation rule")
        protocol = {"format": "locked_protocol_7h_v1", "scope": plan["settings"]["scope"],
            "settings": plan["settings"], "selected_short_budget": choice["selected_short_budget"],
            "validation_reference_target_met": choice["reference_target_met"],
            "test_layout_ids": plan["test_layout_ids"], "test_public_counts": plan["test_public_counts"],
            "cases_per_layout_K": plan["max_cases_per_job"], "models": read(output / "models.json"),
            "source_hashes": plan["source_hashes"], "pso_settings": plan["pso_settings"],
            "certificate_pool_size": plan["certificate_pool_size"], "certificate_alpha_per_group": plan["certificate_alpha_per_group"],
            "code_sha256": plan["code_sha256"], "development_complete_sha256": digest(output / "completion.json"),
            "development_plan_sha256": digest(output / "plan.json"), "methods": list(METHODS),
            "primary_comparisons": [["gnn_short_pso", m] for m in ("random_short_pso", "mlp_short_pso", "full_pso")],
            "in_domain_K": [2, 3, 4, 5], "unseen_K": [6], "test_used_for_selection": False,
            "cv_threshold_source": "Post-hoc train-K2 feature CV threshold0.9, frozen before final test",
            "normalizer_policy": "Original labeled 32-layout train data only; no test refit",
            "fallback": "none", "independent_selection_bank": False,
            "plan_amendment": plan["plan_amendment"],
            "claim_scope": "Matched-budget learned initialization versus random and MLP; no claim that deep learning dominates all simple physical methods"}
        write(output / "protocol.json", protocol, immutable=True)
        write(output / "protocol_receipt.json", {"protocol_sha256":digest(output / "protocol.json"),
              "development_complete_sha256":digest(output / "completion.json")},immutable=True)
        return {"protocol": str(output / "protocol.json"), "sha256": digest(output / "protocol.json"),
                "selected_short_budget": protocol["selected_short_budget"], "scope": protocol["scope"],
                "validation_reference_target_met": protocol["validation_reference_target_met"],
                "final_test_generated": False, "methods": list(METHODS)}


def final_test(protocol_path, output, workers=2, resume=False):
    configure_runtime(0, 1, "cpu")
    protocol_path = Path(protocol_path).resolve()
    development = protocol_path.parent
    output = Path(output).resolve()
    separate(output, development)
    separate(output, ROOT / "step7h")
    if protocol_path.name != "protocol.json": raise ValueError("Use the protocol.json created by the seal command")
    protocol = read(protocol_path)
    # Rebuild the expected protocol from verified development outputs before any final-test parameters/geometry.
    sealed = seal(development)
    if sealed["sha256"] != digest(protocol_path): raise ValueError("Protocol is not the verified development selection")
    receipt = read(development / "protocol_receipt.json")
    if receipt["protocol_sha256"] != digest(protocol_path): raise ValueError("Sealed protocol bytes changed")
    if protocol.get("format") != "locked_protocol_7h_v1" or protocol["test_used_for_selection"] is not False:
        raise ValueError("A sealed 7h protocol is required")
    if protocol["code_sha256"] != code_hashes(): raise ValueError("Code changed after the development protocol was locked")
    if digest(development / "completion.json") != protocol["development_complete_sha256"]:
        raise ValueError("Development completion changed after sealing")
    verify(development, read(development / "completion.json"))
    c = settings(protocol["settings"])
    if type(workers) is not int or not 1 <= workers <= 4: raise ValueError("Use 1..4 CPU workers")
    source = read(development / "inputs/corpus_plan.json")
    if c["scope"] == "main" and (protocol["test_layout_ids"] != source["config"]["reserved_test_layout_ids"]
                                or protocol["test_public_counts"] != [2, 3, 4, 5, 6]):
        raise ValueError("The locked main test layout/K list changed")
    if set(protocol["test_layout_ids"]) & set(source["config"]["training_layout_ids"]+source["config"]["validation_layout_ids"]):
        raise ValueError("Final-test layout overlaps development")
    # Construct parameter schedules only after the protocol has been verified. No test labels are generated.
    jobs = []
    for lid in protocol["test_layout_ids"]:
        cases = parameter_cases(source["config"], source["root_seed"], lid)[:protocol["cases_per_layout_K"]]
        for k in protocol["test_public_counts"]:
            jobs.append({"id": f"test_l{lid:06d}_k{k}", "layout_id": lid, "n_public": k, "split": "test",
                         "regime": "seen_k" if k in protocol["in_domain_K"] else "unseen_k", "cases": cases})
    plan = {"format": FORMAT, "stage": "final", "settings": c, "workers": workers, "jobs": jobs,
        "short_budgets": [protocol["selected_short_budget"]], "pso_settings": protocol["pso_settings"],
        "source_hashes": protocol["source_hashes"], "protocol_sha256": digest(protocol_path),
        "code_sha256": code_hashes(), "versions": versions(),
        "certificate_pool_size": protocol["certificate_pool_size"], "certificate_alpha_per_group": protocol["certificate_alpha_per_group"],
        "test_used_for_selection": False, "source_development_complete_sha256": protocol["development_complete_sha256"],
        "scope": c["scope"], "no_new_reference_labels": True, "no_fallback": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output)+".lock", timeout=0):
        if output.exists() and any(output.iterdir()):
            if not (output / "plan.json").exists() or read(output / "plan.json") != plan:
                raise ValueError("Final output is nonempty or belongs to another protocol; do not retune an opened test")
            if not resume and not (output / "completion.json").exists():
                raise ValueError("Incomplete final run exists; add --resume with the same protocol")
        output.mkdir(parents=True, exist_ok=True)
        write(output / "plan.json", plan, immutable=True)
        snapshot_code(output)
        write(output / "protocol.json", protocol, immutable=True)
        marker = output / "test_opened.json"
        if not marker.exists(): write(marker, {"protocol_sha256": digest(protocol_path), "opened_utc": now(), "scope": c["scope"]}, immutable=True)
        for path in (development / "inputs").iterdir():
            if not path.is_file(): continue
            target = output / "inputs" / path.name
            target.parent.mkdir(exist_ok=True)
            if target.exists() and digest(target) != digest(path): raise ValueError("Copied input changed")
            if not target.exists(): copy2(path, target)
        for ref in protocol["models"].values():
            source_path, target = safe_path(development, ref["path"]), safe_path(output, ref["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            if digest(source_path) != ref["sha256"]: raise ValueError("Locked deployment checkpoint changed")
            if target.exists() and digest(target) != ref["sha256"]: raise ValueError("Copied deployment checkpoint changed")
            if not target.exists(): copy2(source_path, target)
        write(output / "models.json", protocol["models"], immutable=True)
        from .checks import check
        if (output / "completion.json").exists(): return dict(check(output), already_complete=True)
        started = time.perf_counter()
        run_stage(output, plan)
        summarize(output, plan, all_rows(output, plan))
        write(output / "run_timing.json", {"this_invocation_seconds": time.perf_counter()-started,
              "note": "Wall time for this invocation only; resumed invocations are not silently added"})
        completion(output, plan)
        return check(output)


def smoke(corpus, training, diagnostics, output, workers=2, resume=False):
    output = Path(output).resolve()
    c = smoke_settings()
    result = develop(corpus, training, diagnostics, output / "development", c, workers, resume)
    seal(output / "development")
    final = final_test(output / "development/protocol.json", output / "test", workers, resume)
    return {"scope": "smoke", "development": result, "test": final,
            "reserved_5000_5039_generated": False, "note": "Reduced runtime smoke test, not a scientific performance result"}
