"""Portable, read-only verification of case products, call counts and audit decisions."""
from pathlib import Path
import math
import numpy as np

from . import FORMAT
from .common import read, lines, digest, verify, committed, code_hashes, safe_path, METHODS
from .settings import settings
from .benchmark import expected_keys, key, all_rows
from .analysis import grouped, budget_selection
from ris_noma.curve_models import load_model
from ris_noma.multik_graph import fit_normalizer, validate_graph
from ris_noma.multik_core import physical_allocation
from ris_noma.multik_data_core import at_target
from ris_noma.validation import family_constants


def same(a, b):
    if isinstance(a, dict): return isinstance(b, dict) and set(a) == set(b) and all(same(v, b[k]) for k,v in a.items())
    if isinstance(a, (list, tuple)): return isinstance(b, (list, tuple)) and len(a) == len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a, float): return isinstance(b, (float, int)) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
    return a == b


def verify_decision(row, plan, k):
    if not row["returned"]:
        if row["passed"] or row["gated_bob_rate"] != 0: raise ValueError("Missing return was counted as a pass")
        return
    expected = at_target(row, row["xi"])
    if any(row[name] != expected[name] for name in expected): raise ValueError("Acceptance differs from physical checks")
    constants = family_constants(plan["settings"]["audit_samples"], plan["certificate_pool_size"], k,
                                 plan["certificate_alpha_per_group"])
    if not same(constants, row["constants"]): raise ValueError("Statistical certificate differs across methods or roles")
    if not math.isclose(row["dep_lower"], max(0.,row["dep_empirical"]-constants["dep_subtraction"]),abs_tol=1e-12):
        raise ValueError("DEP lower bound was silently changed")
    if not math.isclose(row["dep_lower_margin"],row["dep_lower"]-row["xi"],abs_tol=1e-12):
        raise ValueError("DEP margin changed")


def verify_cv(output, parent):
    output = Path(output)
    plan = read(output / "plan.json")
    verify(output, read(output / "completion.json"))
    if plan["parent_plan_sha256"] != digest(Path(parent) / "plan.json"): raise ValueError("CV parent changed")
    train = lines(output / "inputs/train.jsonl")
    for ref in read(output / "fold_models.json").values():
        lid = ref["held_out_layout_id"]
        ids = sorted({r["layout_id"] for r in train if r["layout_id"] != lid})
        if lid in ref["training_layout_ids"] or ref["training_layout_ids"] != ids: raise ValueError("Held-out layout entered fold training")
        norm = fit_normalizer([r["graph"] for r in train if r["layout_id"] != lid and r["has_label"]])
        if norm != ref["normalization"]: raise ValueError("CV fold normalization leaked held-out data")
        path = safe_path(output,ref["path"])
        if digest(path) != ref["sha256"]: raise ValueError("CV checkpoint changed")
        _, saved = load_model(path)
        if (saved["training_layout_ids"] != ids or saved["epoch"] != plan["settings"]["cv_epochs"]
                or saved["seed"] != plan["settings"]["cv_seed"] or saved["normalization"] != norm):
            raise ValueError("CV checkpoint policy differs from fixed-epoch holdout")
    rows_count = 0
    anchor = read(output / "allocations_frozen.json")
    verify(output, anchor)
    for job in plan["jobs"]:
        prediction = committed(output / f"predictions/{job['id']}.json")
        audited = committed(output / f"audit/{job['id']}.json")
        if audited["allocations_sha256"] != digest(output / "allocations_frozen.json"): raise ValueError("CV audit anchor changed")
        expected = {(f"{job['id']}_c{c['case_index']:03d}", m, d) for c in job["cases"]
            for m in ("gnn_direct", "mlp_direct") for d in (0., .005)}
        actual = {(r["case_id"],r["method"],r["delta"]) for r in audited["rows"]}
        if actual != expected or len(actual) != len(audited["rows"]): raise ValueError("CV evaluation denominator changed")
        lookup = {(r["case_id"],r["method"],r["delta"]):r for r in prediction["rows"]}
        for row in audited["rows"]:
            original = lookup[(row["case_id"],row["method"],row["delta"])]
            if row["allocation"] != original["allocation"]: raise ValueError("CV predictions changed after audit")
            verify_decision(row,plan,job["n_public"])
        rows_count += len(audited["rows"])
    return rows_count


def check(output):
    output = Path(output).resolve()
    plan = read(output / "plan.json")
    if plan.get("format") != FORMAT or plan["code_sha256"] != code_hashes(): raise ValueError("7h code/format changed")
    settings(plan["settings"])
    if not (output / "completion.json").exists():
        return {"complete":False,"stage":plan["stage"],"scope":plan["settings"]["scope"],
            "search_groups_complete":sum(committed(output / f"search/{j['id']}.json") is not None for j in plan["jobs"]),
            "planned_groups":len(plan["jobs"]),"allocations_frozen":(output/"allocations_frozen.json").exists(),
            "audit_groups_complete":sum(committed(output / f"audit/{j['id']}.json") is not None for j in plan["jobs"]),
            "cv_complete":(output/"cv/completion.json").exists()}
    complete = read(output / "completion.json")
    if complete["plan_sha256"] != digest(output / "plan.json"): raise ValueError("Completion plan changed")
    files = verify(output, complete)
    verify(output / "source_snapshot", {"sha256": plan["code_sha256"]})
    train = lines(output / "inputs/train.jsonl")
    validation = lines(output / "inputs/validation.jsonl")
    source = read(output / "inputs/corpus_plan.json")
    norm = fit_normalizer([r["graph"] for r in train if r["has_label"]])
    train_ids = sorted({r["layout_id"] for r in train})
    val_ids = sorted({r["layout_id"] for r in validation})
    if len(train_ids) != 32 or set(train_ids)&set(val_ids): raise ValueError("Train/validation layout leakage")
    if len(train) != 32*4*8 or len(validation) != 16*4*8: raise ValueError("Source case denominator changed")
    models = read(output / "models.json")
    if set(models) != {f"{kind}_s{seed}" for kind in ("gnn","mlp") for seed in (7,17,27)}:
        raise ValueError("Main six small checkpoints are incomplete")
    bounds = read(output / "inputs/physics.json")["base_config"]["bounds"]
    for name, ref in models.items():
        path = safe_path(output, ref["path"])
        if digest(path) != ref["sha256"]: raise ValueError("Deployment checkpoint hash changed")
        _, saved = load_model(path)
        if (saved["normalization"] != norm or saved["training_layout_ids"] != train_ids
                or saved["trained_public_counts"] != [2,3,4,5] or saved["architecture_name"] != "small"
                or saved["epoch"] != ref["epoch"] or ref["delta"] != .005
                or saved["model_kind"] != ref["kind"] or saved["seed"] != ref["seed"]):
            raise ValueError("Main checkpoint lineage, delta or train-only normalizer changed")
    if plan["stage"] == "final":
        protocol = read(output / "protocol.json")
        if digest(output / "protocol.json") != plan["protocol_sha256"] or read(output / "test_opened.json")["protocol_sha256"] != plan["protocol_sha256"]:
            raise ValueError("Final-test opening does not match the locked protocol")
        if protocol["test_used_for_selection"] or set(j["layout_id"] for j in plan["jobs"]) & set(train_ids+val_ids):
            raise ValueError("Final-test selection/layout leakage")
        if plan["settings"]["scope"] == "main" and sorted({j["layout_id"] for j in plan["jobs"]}) != source["config"]["reserved_test_layout_ids"]:
            raise ValueError("Reserved final layout list changed")
    verify(output, read(output / "allocations_frozen.json"))
    verify(output, read(output / "audit_frozen.json"))
    total, objective_calls, observations = 0, 0, []
    for job in plan["jobs"]:
        proposal = committed(output / f"search/{job['id']}.json")
        audited = committed(output / f"audit/{job['id']}.json")
        if (proposal["plan_sha256"] != digest(output / "plan.json")
                or audited["plan_sha256"] != digest(output / "plan.json")
                or audited["allocations_sha256"] != digest(output / "allocations_frozen.json")):
            raise ValueError("Search/audit barriers changed")
        wanted = expected_keys(plan,job)
        if (len(proposal["rows"]) != len(wanted) or {key(r) for r in proposal["rows"]} != wanted
                or len(audited["rows"]) != len(wanted) or {key(r) for r in audited["rows"]} != wanted):
            raise ValueError("Scenario/method/replicate product changed")
        lookup = {key(r):r for r in proposal["rows"]}
        for graph in proposal["graphs"]: validate_graph(graph["graph"])
        calls = 0
        for row in audited["rows"]:
            original = lookup[key(row)]
            if row["allocation"] != original["allocation"]: raise ValueError("Allocation changed after search freeze")
            if row["returned"]:
                y = physical_allocation(row["allocation"], job["n_public"], bounds)
                if not np.allclose(y,row["allocation"],rtol=0,atol=1e-12): raise ValueError("Recorded simplex differs")
                expected_calls = row["budget"]
                history = original["history"]
                if expected_calls:
                    if (len(history) != expected_calls//30 or history[-1]["evaluations"] != expected_calls
                            or [h["evaluations"] for h in history] != list(range(30,expected_calls+1,30))):
                        raise ValueError("PSO sweep/objective budget mismatch")
                    if expected_calls >= 90 and (history[1]["inertia_used_for_update"] != .90
                            or not math.isclose(history[-1]["inertia_used_for_update"], .40, abs_tol=1e-12)):
                        raise ValueError("Budget-specific inertia schedule changed")
                elif history: raise ValueError("Direct prediction has a hidden PSO search")
                expected_time = sum(original[n] for n in ("input_bank_amortized_seconds","graph_seconds","forward_seconds",
                    "correction_seconds","objective_setup_seconds","pso_seconds"))
                if not math.isclose(expected_time,row["online_seconds"],abs_tol=1e-12): raise ValueError("Online timing omitted a stage")
            else: expected_calls = 0
            if row["objective_evaluations"] != expected_calls: raise ValueError("Reported objective count differs from actual budget")
            verify_decision(row,plan,job["n_public"])
            calls += row["objective_evaluations"]
        if calls != proposal["objective_evaluations"]: raise ValueError("Job objective total differs")
        objective_calls += calls
        total += len(audited["rows"])
        observations.extend(audited["rows"])
    expected_metrics = grouped(observations,("regime","method","budget"))
    summary = read(output / "summary.json") if (output / "summary.json").exists() else None
    if summary is not None:
        if not same(expected_metrics, summary["metrics"]) or summary["objective_evaluations"] != objective_calls:
            raise ValueError("Summary does not reproduce from committed physical rows")
    cv_rows = 0
    if plan["stage"] == "development":
        if budget_selection(observations,plan["settings"]) != read(output / "budget_selection.json"):
            raise ValueError("Budget selection is not validation-only under the fixed rule")
        cv_rows = verify_cv(output / "cv", output)
    return {"step":"7h","complete":True,"stage":plan["stage"],"scope":plan["settings"]["scope"],
        "methods":list(METHODS),"layouts":len({j["layout_id"] for j in plan["jobs"]}),"groups":len(plan["jobs"]),
        "physical_rows":total,"cv_physical_rows":cv_rows,"objective_evaluations":objective_calls,
        "verified_files":files,"test_used_for_selection":False,"final_test_generated":plan["stage"]=="final",
        "reserved_5000_5039_generated":plan["stage"]=="final" and plan["settings"]["scope"]=="main"}
