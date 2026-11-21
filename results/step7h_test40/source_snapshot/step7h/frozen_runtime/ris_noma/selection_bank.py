"""6g immutable protocol: search -> independent selection -> independent audit.

No training, repair, extra PSO evaluation, or audit-based decision is performed.
Abstentions remain in all denominators. One validation-frozen 6f GNN is reused.
"""
from copy import deepcopy
from itertools import product
from pathlib import Path
import platform
import shutil
import time

from filelock import FileLock
import numpy as np
import scipy
import torch

from .budget_benchmark import new_layout, record_path, run_phase
from .dep_label_experiment import channel_context, freeze_phase
from .dep_label_learning import score_cases, write_immutable
from .expanded_data import save_json, write_table
from .expanded_scenarios import ROOT
from .learning_data import read_json, sha256
from .physics_training import CODE_FILES as PREVIOUS_CODE, check_experiment as check_6f


STREAMS = {"validation_search": 43, "validation_pso": 44, "validation_jitter": 45,
           "validation_select": 46, "validation_audit": 47, "test_search": 48,
           "test_pso": 49, "test_jitter": 50, "test_select": 51, "test_audit": 52, "bootstrap": 53}
CODE_FILES = tuple(dict.fromkeys((*PREVIOUS_CODE, "run_step6g.py", "ris_noma/candidate_archive.py",
    "ris_noma/selection_bank.py", "ris_noma/selection_bank_search.py", "ris_noma/selection_bank_analysis.py")))


def validate_config(config, swarm, xis):
    fields = {"test_layout_ids", "budgets", "primary_budget", "full_evaluations", "search_margins",
              "archive_slacks", "candidate_limit", "search_samples", "selection_samples", "audit_samples",
              "bootstrap_repetitions"}
    if set(config) != fields:
        raise ValueError("Invalid step 6g config fields")
    ids, budgets = config["test_layout_ids"], config["budgets"]
    if (not isinstance(ids, list) or len(ids) < 2 or ids != sorted(set(ids))
            or any(type(i) is not int or not 600 <= i <= 9999 for i in ids)):
        raise ValueError("Use at least two distinct new test IDs in [600,9999]")
    if (not isinstance(budgets, list) or not budgets or budgets != sorted(set(budgets))
            or any(type(b) is not int or b < swarm or b % swarm for b in budgets)
            or type(config["primary_budget"]) is not int or config["primary_budget"] not in budgets
            or type(config["full_evaluations"]) is not int or config["full_evaluations"] <= max(budgets)
            or config["full_evaluations"] % swarm):
        raise ValueError("Declare whole-swarm budgets, a primary budget and a larger full-PSO budget")
    for name in ("search_margins", "archive_slacks"):
        values = config[name]
        if (not isinstance(values, list) or not values or values != sorted(set(values))
                or any(type(v) not in (int, float) or not np.isfinite(v) or v < 0 for v in values)):
            raise ValueError(f"Invalid ordered margins: {name}")
    if (len(config["archive_slacks"]) != 2 or config["candidate_limit"] != 5
            or max(xis) + max(config["search_margins"]) + max(config["archive_slacks"]) >= 1):
        raise ValueError("6g uses a five-slot archive, two extra slacks and thresholds below one")
    for name in ("search_samples", "selection_samples", "audit_samples", "bootstrap_repetitions"):
        if type(config[name]) is not int or config[name] < 1000:
            raise ValueError(f"{name} must be an integer >= 1000")
    return deepcopy(config)


def source_fingerprints(previous, reference):
    return {name: sha256(previous / name) for name in (
        "completion.json", "plan.json", "selection_frozen.json", "source_physics_plan.json", reference["path"])}


def rows_per_case(plan, role):
    return 2 + 4 * len(plan["budgets"]) * len(plan["search_config"]["pso_seeds"]) + (
        2 * len(plan["search_config"]["pso_seeds"]) if role == "test" else 0)


def inspect_experiment(previous, config, workers):
    previous = Path(previous).resolve()
    check_6f(previous)
    old = read_json(previous / "plan.json")
    selection = read_json(previous / "selection_frozen.json")
    source = read_json(previous / "source_physics_plan.json")
    config = validate_config(config, old["search_config"]["swarm_size"], old["xi"])
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    model_name = selection["representatives"][selection["chosen_group"]]
    reference = selection["selected"][model_name]
    if sha256(previous / reference["path"]) != reference["sha256"] or selection["test_used_for_selection"]:
        raise ValueError("Need the unmodified, validation-selected 6f GNN")
    seen = set(old["training_layout_ids"] + old["validation_layout_ids"] + old["test_layout_ids"])
    if seen.intersection(config["test_layout_ids"]):
        raise ValueError("6g test layouts must be distinct from 6f train/validation/test")
    powers, xis = old["power_dbm"], old["xi"]
    validation, test = old["validation_layouts"], [dict(new_layout(source, lid), split="test") for lid in config["test_layout_ids"]]
    seeds, cap = len(old["search_config"]["pso_seeds"]), config["candidate_limit"]
    alpha = old["search_config"]["audit_alpha"]
    plan = {"step": "6g", **config, "workers": workers, "streams": STREAMS,
        "selected_model": {"name": model_name, "source_reference": reference, "path": "model/gnn.pt",
                           "sha256": reference["sha256"], "seed": reference["seed"], "epoch": reference["epoch"]},
        "sources_sha256": source_fingerprints(previous, reference), "search_config": old["search_config"],
        "power_dbm": powers, "xi": xis, "validation_layouts": validation, "test_layouts": test,
        "validation_layout_ids": [e["layout_id"] for e in validation], "feature_schema": old["feature_schema"],
        "search_samples_per_layout": config["search_samples"], "audit_samples_per_layout": config["audit_samples"],
        "test_used_for_tuning": False, "gnn_retrained": False,
        "scope": "K=2, fixed Alice/RIS; same receiver-position distribution, new test IDs; one frozen GNN",
        "selection_rule": "max selection-bank Bob rate among all bound-passing candidates; then larger DEP lower bound and candidate ID; abstain if none",
        "margin_rule": "pooled validation audit passes of cold/warm screened methods at primary budget; gated rate; smaller common margin",
        "timing_scope": "online: graph/inference, observed PSO prefix, complete selection-pool evaluation; "
                        "standalone estimate additionally charges complete search/selection bank generation and PSO cache per decision; "
                        "offline train/model load/warmup and final research audit excluded",
        "code_sha256": {name: sha256(ROOT / name) for name in CODE_FILES},
        "versions": {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version(),
                     "numpy": np.__version__, "scipy": scipy.__version__, "torch": str(torch.__version__)}}
    for role, entries, margins in (("validation", validation, len(config["search_margins"])), ("test", test, 1)):
        per_case = rows_per_case(plan, role)
        pools_per_seed = 2 * len(config["budgets"]) + (1 if role == "test" else 0)
        plan[f"{role}_selection_family_size"] = len(xis) * margins * (1 + seeds * pools_per_seed * cap)
        plan[f"{role}_audit_family_size"] = len(xis) * margins * per_case
        plan[f"{role}_alpha_per_family"] = alpha / (len(entries) * len(powers))
        plan[f"{role}_audit_rows_planned"] = len(entries) * len(powers) * len(xis) * margins * per_case
        per_seed_evaluations = 2 * max(config["budgets"]) + (config["full_evaluations"] if role == "test" else 0)
        plan[f"{role}_pso_evaluations_planned"] = len(entries) * len(powers) * len(xis) * margins * seeds * per_seed_evaluations
    return plan, source


def choose_candidate(scored):
    if not scored or len({c["candidate_id"] for c in scored}) != len(scored):
        raise ValueError("Need a nonempty pool with distinct candidate IDs")
    accepted = [c for c in scored if c["audit_pass"]]
    return min(accepted, key=lambda c: (-c["bob_rate"], -c["dep_lower"], c["candidate_id"])) if accepted else None


def select_layout(entry, plan, source, search, role, freeze):
    bank_started = time.perf_counter()
    base, layout, bank = channel_context(plan, source, entry, f"{role}_select", plan["selection_samples"])
    bank_seconds = time.perf_counter() - bank_started
    order = np.asarray(search["order"], int)
    decisions, pools, unique = [], [], 0
    for pool in search["pools"]:
        started = time.perf_counter()
        candidates = [{"candidate_id": c["candidate_id"], "allocation": c["allocation"],
                       "total_power_dbm": pool["total_power_dbm"], "xi": pool["xi"]} for c in pool["candidates"]]
        scored, count = score_cases(base, layout, bank, order, candidates,
                                    plan[f"{role}_selection_family_size"], plan[f"{role}_alpha_per_family"])
        score_seconds = time.perf_counter() - started
        unique += count
        winner = choose_candidate(scored)
        raw = next(c for c in pool["candidates"] if c["candidate_id"] == pool["raw_candidate_id"])
        identity = {key: pool[key] for key in ("layout_id", "total_power_dbm", "xi", "search_margin", "branch", "budget", "pso_seed")}
        common = {**identity, "search_wall_seconds": pool["search_wall_seconds"],
                  "search_bank_seconds": pool["search_bank_seconds"], "objective_cache_seconds": pool["objective_cache_seconds"],
                  "evaluations": pool["budget"], "candidate_count": len(scored)}
        for policy, chosen in (("raw", raw), ("screened", winner)):
            screened = policy == "screened"
            evaluation_seconds = score_seconds if screened else 0.
            selection_bank_seconds = bank_seconds if screened else 0.
            online = pool["search_wall_seconds"] + evaluation_seconds
            decisions.append({**common, "method": f"{pool['branch']}_{policy}", "policy": policy,
                "allocation": None if chosen is None else chosen["allocation"],
                "candidate_id": None if chosen is None else chosen["candidate_id"],
                "abstained": chosen is None, "selection_pool_seconds": evaluation_seconds,
                "selection_bank_seconds": selection_bank_seconds,
                "online_seconds": online,
                "standalone_estimate_seconds": online + pool["search_bank_seconds"] + pool["objective_cache_seconds"] + selection_bank_seconds})
        pools.append({**identity, "candidates": scored, "selected_candidate_id": None if winner is None else winner["candidate_id"],
                      "selection_pool_seconds": score_seconds, "selection_bank_seconds": bank_seconds})
    return {"layout_id": entry["layout_id"], "candidates_frozen_sha256": freeze, "rows": decisions,
            "selection_pools": pools, "selection_candidate_evaluations": unique}


def audit_layout(entry, plan, source, decision, order, role, freeze):
    base, layout, bank = channel_context(plan, source, entry, f"{role}_audit", plan["audit_samples"])
    returned = [r for r in decision["rows"] if not r["abstained"]]
    audited, count = score_cases(base, layout, bank, np.asarray(order, int), returned,
                               plan[f"{role}_audit_family_size"], plan[f"{role}_alpha_per_family"])
    rows = [*audited]
    for row in decision["rows"]:
        if row["abstained"]:
            rows.append({**row, "audit_pass": False, "bob_rate": 0., "gated_bob_rate": 0.,
                         "dep_gamma": None, "dep_empirical": None, "dep_lower": None,
                         "bob_outage_upper": None, "qos_lower_margin": None,
                         "failed_checks": ["selection_rejected_all"]})
    return {"layout_id": entry["layout_id"], "decisions_frozen_sha256": freeze,
            "rows": rows, "unique_audit_evaluations": count}


def run_stage(output, plan, source, role, margins, model, upstream):
    from .selection_bank_search import search_layout
    entries = plan[f"{role}_layouts"]
    phase = f"{role}_candidates"
    searches = run_phase(output, phase, search_layout,
        [(e, plan, source, role, margins, model, upstream) for e in entries], plan["workers"],
        (output / f"{phase}_frozen.json").is_file())
    if any(r["upstream_frozen_sha256"] != upstream for r in searches.values()):
        raise ValueError("Candidate pools belong to another frozen upstream")
    candidate_hash = freeze_phase(output, phase, searches, f"{phase}_frozen.json")
    eligible = [e for e in entries if searches[e["layout_id"]]["status"] == "eligible"]
    if not eligible:
        raise ValueError(f"No eligible {role} layout")
    print(f"{role}: all candidate pools frozen before opening the selection bank", flush=True)
    phase = f"{role}_decisions"
    decisions = run_phase(output, phase, select_layout,
        [(e, plan, source, searches[e["layout_id"]], role, candidate_hash) for e in eligible], plan["workers"],
        (output / f"{phase}_frozen.json").is_file())
    if any(r["candidates_frozen_sha256"] != candidate_hash for r in decisions.values()):
        raise ValueError("Decisions belong to different candidates")
    decision_hash = freeze_phase(output, phase, decisions, f"{phase}_frozen.json")
    print(f"{role}: all final decisions frozen before opening the audit bank", flush=True)
    phase = f"{role}_audit"
    audits = run_phase(output, phase, audit_layout,
        [(e, plan, source, decisions[e["layout_id"]], searches[e["layout_id"]]["order"], role, decision_hash)
         for e in eligible], plan["workers"], (output / f"{phase}_frozen.json").is_file())
    if any(r["decisions_frozen_sha256"] != decision_hash for r in audits.values()):
        raise ValueError("Audit belongs to different frozen decisions")
    freeze_phase(output, phase, audits, f"{phase}_frozen.json")
    rows = [row for lid in sorted(audits) for row in audits[lid]["rows"]]
    expected = len(eligible) * len(plan["power_dbm"]) * len(plan["xi"]) * len(margins) * rows_per_case(plan, role)
    keys = {(r["layout_id"], r["total_power_dbm"], r["xi"], r["search_margin"], r["method"], r["budget"], r["pso_seed"]) for r in rows}
    if len(rows) != expected or len(keys) != expected:
        raise ValueError("Missing or duplicated eligible case/method/seed/budget")
    evaluated = sum(r["pso_evaluations"] for r in searches.values())
    scale = plan[f"{role}_pso_evaluations_planned"] / len(entries)
    if evaluated != len(eligible) * scale:
        raise ValueError("PSO evaluation count differs from declared paired trajectories")
    return rows, {"eligible_layout_ids": [e["layout_id"] for e in eligible],
        "excluded_layouts": {lid: r["status"] for lid, r in searches.items() if r["status"] != "eligible"},
        "pso_evaluations": evaluated, "audit_rows": len(rows),
        "selection_candidate_evaluations": sum(r["selection_candidate_evaluations"] for r in decisions.values()),
        "unique_audit_evaluations": sum(r["unique_audit_evaluations"] for r in audits.values())}


def freeze_margin(output, plan, rows):
    scores = []
    for margin in plan["search_margins"]:
        group = [r for r in rows if r["search_margin"] == margin and r["budget"] == plan["primary_budget"]
                 and r["method"] in ("cold_screened", "warm_screened")]
        scores.append({"search_margin": margin, "runs": len(group), "passed": sum(r["audit_pass"] for r in group),
                       "gated_bob_rate": float(np.mean([r["gated_bob_rate"] for r in group])),
                       "abstentions": sum(r["abstained"] for r in group)})
    chosen = min(scores, key=lambda r: (-r["passed"], -r["gated_bob_rate"], r["search_margin"]))["search_margin"]
    frozen = {"chosen_search_margin": chosen, "validation_audit_frozen_sha256": sha256(output / "validation_audit_frozen.json"),
              "scores": scores, "test_used_for_selection": False, "primary_budget": plan["primary_budget"]}
    write_immutable(output / "selection_frozen.json", frozen)
    write_table(output / "validation_margins.csv", scores)
    write_table(output / "validation_runs.csv", rows)
    return frozen


def check_experiment(output):
    output = Path(output).resolve()
    hashes = read_json(output / "completion.json")["sha256"]
    required = {"plan.json", "source_physics_plan.json", "model/gnn.pt", "selection_frozen.json", "summary.json",
                "comparison_runs.csv", "comparison_by_method.csv", "paired_comparison.csv"}
    if not required.issubset(hashes):
        raise ValueError("Incomplete 6g output")
    for name, digest in hashes.items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"6g result changed: {name}")
    plan, result, selection = (read_json(output / name) for name in ("plan.json", "summary.json", "selection_frozen.json"))
    if (result["step"] != "6g" or not result["complete"] or selection["test_used_for_selection"]
            or result["chosen_search_margin"] != selection["chosen_search_margin"]
            or result["test_cases"] != result["test_layouts"] * len(plan["power_dbm"]) * len(plan["xi"])
            or result["test_audit_rows"] != result["test_cases"] * rows_per_case(plan, "test")
            or result["test_pso_evaluations"] != result["test_layouts"] * plan["test_pso_evaluations_planned"] / len(plan["test_layouts"])):
        raise ValueError("6g counts or selection differ from the frozen plan")
    for role in ("validation", "test"):
        for stage in ("candidates", "decisions", "audit"):
            name = f"{role}_{stage}_frozen.json"
            if name not in hashes or any(hashes.get(p) != h for p, h in read_json(output / name)["sha256"].items()):
                raise ValueError(f"Incomplete or changed phase: {name}")
    return result


def run_experiment(args, plan, source):
    from .selection_bank_analysis import report
    output, previous = Path(args.output).resolve(), Path(args.previous).resolve()
    if output == previous or output in previous.parents or previous in output.parents:
        raise ValueError("Source and output must be separate, non-nested directories")
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output.with_name(f".{output.name}.lock")), timeout=0):
        if output.exists() and any(output.iterdir()):
            if not args.resume or not (output / "plan.json").is_file() or read_json(output / "plan.json") != plan:
                raise ValueError("Use a new output or --resume with the same inputs, code, versions and worker count")
        output.mkdir(exist_ok=True)
        if (output / "completion.json").is_file():
            return check_experiment(output)
        write_immutable(output / "plan.json", plan)
        write_immutable(output / "source_physics_plan.json", source)
        (output / "model").mkdir(exist_ok=True)
        model = output / plan["selected_model"]["path"]
        if not model.exists():
            shutil.copyfile(previous / plan["selected_model"]["source_reference"]["path"], model)
        if sha256(model) != plan["selected_model"]["sha256"]:
            raise ValueError("Copied GNN checkpoint differs from frozen 6f source")
        val_rows, val_info = run_stage(output, plan, source, "validation", plan["search_margins"], model, sha256(output / "plan.json"))
        selection = freeze_margin(output, plan, val_rows)
        print(f"One common search margin frozen on validation: {selection['chosen_search_margin']:g}", flush=True)
        rows, test_info = run_stage(output, plan, source, "test", [selection["chosen_search_margin"]], model,
                                   sha256(output / "selection_frozen.json"))
        if source_fingerprints(previous, plan["selected_model"]["source_reference"]) != plan["sources_sha256"]:
            raise ValueError("Source 6f files changed during 6g")
        result = report(output, plan, selection, rows, val_info, test_info)
        save_json(output / "summary.json", result)
        save_json(output / "completion.json", {"sha256": {
            p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*"))
            if p.is_file() and p.suffix != ".tmp" and p != output / "completion.json" and (output / "figures") not in p.parents}})
        return check_experiment(output)
