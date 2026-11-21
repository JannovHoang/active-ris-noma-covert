"""Step 6a PSO label generation, independent audit, and grouped graph export."""
import csv
import json
import os
from pathlib import Path
import time

import numpy as np

from .channels import build_bank, public_order
from .config import dbm_to_w
from .cross_xi import select_across_xi, target_checks
from .dataset import SPLITS, check_dataset, pack_graphs
from .expanded_scenarios import FORMAT, physical_config, reproduce_layout, scenario_rng
from .graph_data import (
    EDGE_FEATURES, GLOBAL_FEATURES, NODE_FEATURES, SCHEMA_VERSION,
    fit_normalizer, original_allocation, statistical_graph,
)
from .learning_data import read_json, sha256
from .objective import CachedObjective
from .pso import PSOSettings, optimize
from .validation import assess_proposal


SEARCH = "proposals_label.json"
SELECTION = "selection_validation.json"
AUDIT = "audit_heldout.json"


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def commit(path, value):
    save_json(path, value)
    save_json(path.with_suffix(".receipt.json"), {"sha256": sha256(path)})


def read_committed(path):
    receipt = path.with_suffix(".receipt.json")
    if not receipt.exists():
        return None
    if not path.is_file() or read_json(receipt)["sha256"] != sha256(path):
        raise ValueError(f"Committed record changed: {path}")
    return read_json(path)


def layout_folder(output, entry):
    return output / "layouts" / f"layout_{entry['layout_id']:06d}"


def make_bank(plan, entry, role):
    experiment = plan["experiment"]
    config = physical_config(plan, entry, experiment["total_power_dbm"][0], experiment[f"{role}_samples"])
    layout = reproduce_layout(config, entry)
    extra = () if role == "label" else (experiment[f"{role}_seed"],)
    bank = build_bank(config, layout, scenario_rng(config, entry["layout_id"], role, *extra))
    return config, layout, bank


def search_layout(plan, entry):
    started = time.perf_counter()
    result = {
        "layout_id": entry["layout_id"], "status": "invalid_geometry",
        "sic_order_original_ids": None, "bob_to_strongest_public_gain": None,
        "pools": [], "pso_evaluations": 0,
    }
    if entry["geometry_valid"]:
        config, layout, bank = make_bank(plan, entry, "label")
        order = public_order(bank)
        means = bank.signal_gain_tilde.mean(axis=0)
        result["sic_order_original_ids"] = order.tolist()
        result["bob_to_strongest_public_gain"] = float(means[0] / max(means[1:-1]))
        objective = CachedObjective(config, layout, bank, order)
        result["status"] = "eligible" if objective.layout_eligible else "bob_not_strongest"
        if objective.layout_eligible:
            experiment = plan["experiment"]
            settings = PSOSettings(**plan["pso_settings"])
            xis = config["requirements"]["xi"]
            margin = experiment["search_dep_margin"]
            for power in experiment["total_power_dbm"]:
                physical = dict(config, total_dbm=power)
                objective = CachedObjective(physical, layout, bank, order)
                graph = statistical_graph(physical, layout, bank, order, xis[0])
                proposals = []
                for index, xi in enumerate(xis):
                    for seed in experiment["pso_seeds"]:
                        target = xi + margin
                        rng = scenario_rng(physical, entry["layout_id"], "pso", seed, index)
                        best = optimize(lambda z: objective.evaluate(z, target), 3, settings, rng)
                        proposals.append({
                            "proposal_id": f"p{len(proposals):03d}",
                            "xi": xi, "margin": margin, "gamma_target": target, "seed": seed,
                            "label_feasible": best.outcome.feasible,
                            "label_metrics": best.outcome.metrics,
                            "normalized_position": best.position.tolist(),
                            "violation": best.outcome.violation,
                            "evaluations": best.evaluations,
                        })
                result["pools"].append({
                    "total_power_dbm": power, "proposals": proposals,
                    "raw_graph": {key: array.tolist() for key, array in graph.items()},
                })
                result["pso_evaluations"] += sum(p["evaluations"] for p in proposals)
    result["seconds"] = time.perf_counter() - started
    return result


def select_layout(plan, entry, search, search_hash):
    started = time.perf_counter()
    result = {"search_sha256": search_hash, "pools": []}
    if search["status"] == "eligible":
        config, layout, bank = make_bank(plan, entry, "validation")
        order = np.asarray(search["sic_order_original_ids"], dtype=int)
        alpha = plan["alpha_per_power_layout_stage"]
        for pool in search["pools"]:
            power = pool["total_power_dbm"]
            physical = dict(config, total_dbm=power)
            candidates = [p for p in pool["proposals"] if p["label_feasible"]]
            assessments = [
                assess_proposal(physical, layout, bank, order, p, len(pool["proposals"]), alpha, True)
                for p in candidates
            ]
            selection = select_across_xi(physical, order, candidates, assessments, config["requirements"]["xi"])
            result["pools"].append({"total_power_dbm": power, "assessments": assessments, "selection": selection})
    result["seconds"] = time.perf_counter() - started
    return result


def frozen_manifest(output, plan):
    jobs = {}
    for entry in plan["layouts"]:
        folder = layout_folder(output, entry)
        search = read_committed(folder / SEARCH)
        selection = read_committed(folder / SELECTION)
        if search is None or selection is None or selection["search_sha256"] != sha256(folder / SEARCH):
            raise ValueError(f"Missing or changed validation selection in {folder}")
        jobs[str(entry["layout_id"])] = {"search": sha256(folder / SEARCH), "selection": sha256(folder / SELECTION)}
    return {"plan_sha256": sha256(output / "plan.json"), "layouts": jobs}


def audit_layout(plan, entry, search, selection, frozen_hash):
    started = time.perf_counter()
    experiment = plan["experiment"]
    xis = plan["base_config"]["requirements"]["xi"]
    order = np.asarray(search["sic_order_original_ids"], dtype=int) if search["status"] == "eligible" else None
    pools = {p["total_power_dbm"]: p for p in search["pools"]}
    choices = {p["total_power_dbm"]: p for p in selection["pools"]}
    have_winners = any(s["proposal_id"] is not None for p in choices.values() for s in p["selection"])
    if have_winners:
        config, layout, bank = make_bank(plan, entry, "heldout")
    cases = []
    for power in experiment["total_power_dbm"]:
        pool = pools.get(power, {})
        proposals = {p["proposal_id"]: p for p in pool.get("proposals", [])}
        choice = choices.get(power, {})
        picked = {s["xi"]: s for s in choice.get("selection", [])}
        validation = {a["proposal_id"]: a for a in choice.get("assessments", [])}
        winners = sorted({s["proposal_id"] for s in picked.values() if s["proposal_id"] is not None})
        heldout = {}
        if winners:
            physical = dict(config, total_dbm=power)
            for pid in winners:
                heldout[pid] = assess_proposal(
                    physical, layout, bank, order, proposals[pid], len(winners),
                    plan["alpha_per_power_layout_stage"], True,
                )
        for xi in xis:
            selected = picked.get(xi, {})
            pid = selected.get("proposal_id")
            status = "no_validation_label" if search["status"] == "eligible" else search["status"]
            row = {
                "layout_id": entry["layout_id"], "split": entry["split"], "total_power_dbm": power, "xi": xi,
                "layout_eligible": search["status"] == "eligible", "status": status, "proposal_id": pid,
                "label_available": pid is not None, "heldout_bound_checks_pass": None,
                "heldout_used_for_label_filter": False,
            }
            if pid is not None:
                proposal, va, ho = proposals[pid], validation[pid], heldout[pid]
                label_metrics = proposal["label_metrics"]
                y, z = original_allocation(label_metrics["allocation_sic_order"], order, physical["bounds"])
                failed = [key for key, ok in target_checks(physical, order, proposal, ho, xi).items() if not ok]
                row.update(
                    status="selected_on_validation", allocation_original_order=y.tolist(),
                    allocation_pso_sic=z.tolist(), sic_order_original_ids=order.tolist(),
                    source_xi=selected["source_xi"], source_margin=selected["source_margin"],
                    pso_seed=proposal["seed"], label_metrics=label_metrics,
                    bob_rate_label=label_metrics["bob_mean_rate_bit_s_hz"],
                    dep_lower_validation=va["screen"]["dep_uniform_lower"],
                    bob_rate_heldout=ho["metrics"]["bob_mean_rate_bit_s_hz"],
                    dep_empirical_heldout=ho["screen"]["dep_empirical_min"],
                    dep_lower_heldout=ho["screen"]["dep_uniform_lower"],
                    bob_outage_upper_heldout=ho["screen"]["bob_outage_upper"],
                    public_rate_lower_heldout_sic_order=ho["screen"]["public_rate_lower_sic_order"],
                    heldout_bound_checks_pass=not failed, failed_heldout_checks=failed,
                )
            cases.append(row)
    return {
        "frozen_sha256": frozen_hash, "cases": cases, "heldout_bank_generated": have_winners,
        "seconds": time.perf_counter() - started,
    }


def case_key(row):
    return row["layout_id"], row["total_power_dbm"], row["xi"]


def export_graphs(output, plan, cases, search_by_layout):
    items = {role: [] for role in SPLITS}
    for row in cases:
        if row["label_available"]:
            search = search_by_layout[row["layout_id"]]
            pool = next(p for p in search["pools"] if p["total_power_dbm"] == row["total_power_dbm"])
            graph = {key: np.asarray(value) for key, value in pool["raw_graph"].items()}
            graph["global_raw"][GLOBAL_FEATURES.index("xi")] = row["xi"]
            graph.update(
                y=np.asarray(row["allocation_original_order"]), z=np.asarray(row["allocation_pso_sic"]),
                order=np.asarray(row["sic_order_original_ids"]), layout_id=row["layout_id"],
                xi=row["xi"], rate_label=row["bob_rate_label"], total_power_dbm=row["total_power_dbm"],
            )
            items[row["split"]].append(graph)
    if not items["train"]:
        return False

    output.mkdir(parents=True, exist_ok=True)
    normalizer = fit_normalizer(np.stack([g["node_raw"] for g in items["train"]]),
                                np.stack([g["global_raw"] for g in items["train"]]))
    normalizer["training_layout_ids"] = sorted({g["layout_id"] for g in items["train"]})
    save_json(output / "normalization.json", normalizer)
    save_json(output / "feature_schema.json", {
        "version": SCHEMA_VERSION, "dataset_format": FORMAT, "case_key": plan["case_key"],
        "node_features": NODE_FEATURES, "global_features": GLOBAL_FEATURES, "edge_features": EDGE_FEATURES,
        "node_order": ["Bob", "public_original_0", "public_original_1"],
        "target_order": ["mb", "public_original_0", "public_original_1", "s"],
        "model_inputs": ["node", "global_features", "edge_index", "edge_attr"],
    })
    with (output / "records.jsonl").open("w", encoding="utf-8") as handle:
        for row in cases:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    for role in SPLITS:
        arrays = pack_graphs(items[role], normalizer)
        arrays["total_power_dbm"] = np.asarray([g["total_power_dbm"] for g in items[role]], dtype=float)
        np.savez_compressed(output / f"{role}.npz", **arrays)
    names = ["normalization.json", "feature_schema.json", "records.jsonl", *[f"{role}.npz" for role in SPLITS]]
    save_json(output / "dataset_manifest.json", {
        "schema": FORMAT, "graph_feature_schema": SCHEMA_VERSION, "partition": plan["partition"],
        "bounds": plan["base_config"]["bounds"], "case_key": plan["case_key"],
        "counts": {role: count_cases([row for row in cases if row["split"] == role]) for role in SPLITS},
        "base_physics_config": plan["base_config"], "label_generation_policy": plan["experiment"],
        "layout_plan": plan["layouts"], "rng_streams": plan["streams"],
        "plan_sha256": sha256(output.parent / "plan.json"),
        "frozen_sha256": sha256(output.parent / "selections_frozen.json"),
        "output_sha256": {name: sha256(output / name) for name in names},
        "label_filter": plan["label_policy"], "scope": "development pilot",
    })
    check_graphs(output, plan, cases)
    return True


def check_graphs(output, plan, cases):
    check_dataset(output)
    manifest = read_json(output / "dataset_manifest.json")
    schema = read_json(output / "feature_schema.json")
    if manifest["schema"] != FORMAT or schema["version"] != SCHEMA_VERSION or manifest["partition"] != plan["partition"]:
        raise ValueError("Graph format or partition differs from the 6a plan")
    ledger = [json.loads(line) for line in (output / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    if ledger != cases:
        raise ValueError("Graph ledger differs from the committed case results")
    p_index = GLOBAL_FEATURES.index("log10_total_power_w")
    for role in SPLITS:
        expected = {case_key(row): row for row in cases if row["split"] == role and row["label_available"]}
        with np.load(output / f"{role}.npz", allow_pickle=False) as data:
            keys = list(zip(data["layout_id"], data["total_power_dbm"], data["xi"]))
            if len(set(keys)) != len(keys) or set(keys) != set(expected):
                raise ValueError(f"Scenario keys differ from labeled cases in {role}")
            if not np.allclose(data["global_raw"][:, p_index], np.log10(dbm_to_w(data["total_power_dbm"])),
                               rtol=0, atol=1e-12):
                raise ValueError(f"Power input differs from the case identity in {role}")
            for i, key in enumerate(keys):
                row = expected[key]
                if not np.array_equal(data["y_allocation_original_order"][i], row["allocation_original_order"]):
                    raise ValueError(f"Allocation differs from the selected PSO label for {key}")


def count_cases(rows):
    eligible = sum(row["layout_eligible"] for row in rows)
    labeled = sum(row["label_available"] for row in rows)
    passed = sum(row["heldout_bound_checks_pass"] is True for row in rows)
    return {
        "planned": len(rows), "eligible": eligible, "labeled": labeled, "missing": len(rows) - labeled,
        "heldout_pass": passed, "heldout_review": labeled - passed,
    }


def write_table(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_cases(plan, cases):
    expected = {
        (entry["layout_id"], power, xi): entry["split"]
        for entry in plan["layouts"] for power in plan["experiment"]["total_power_dbm"]
        for xi in plan["base_config"]["requirements"]["xi"]
    }
    actual = {case_key(row): row["split"] for row in cases}
    if actual != expected or len(cases) != len(expected):
        raise ValueError("Case ledger does not cover the declared layout/power/DEP grid")


def summarize(output, plan, searches, selections, audits):
    started = time.perf_counter()
    cases = [row for audit in audits for row in audit["cases"]]
    check_cases(plan, cases)
    by_layout = {search["layout_id"]: search for search in searches}
    graph_exported = export_graphs(output / "graphs", plan, cases, by_layout)
    layouts = []
    for entry in plan["layouts"]:
        search = by_layout[entry["layout_id"]]
        public = entry["public_xy_original_order"] or [[None, None], [None, None]]
        coordinates = {**entry["positions"], "public_0": public[0], "public_1": public[1]}
        layouts.append({
            "layout_id": entry["layout_id"], "split": entry["split"], "status": search["status"],
            "bob_to_strongest_public_gain": search["bob_to_strongest_public_gain"],
            **{f"{node}_{axis}_m": xy[i] for node, xy in coordinates.items() for i, axis in enumerate(("x", "y"))},
        })
    write_table(output / "layouts.csv", layouts)
    fields = (
        "layout_id", "split", "total_power_dbm", "xi", "status", "proposal_id",
        "bob_rate_label", "bob_rate_heldout", "dep_lower_validation", "dep_lower_heldout",
        "heldout_bound_checks_pass", "failed_heldout_checks",
    )
    write_table(output / "cases.csv", [{key: row.get(key) for key in fields} for row in cases])
    coverage = []
    for power in plan["experiment"]["total_power_dbm"]:
        for xi in plan["base_config"]["requirements"]["xi"]:
            group = [row for row in cases if row["total_power_dbm"] == power and row["xi"] == xi]
            rates = [row["bob_rate_heldout"] for row in group if row["heldout_bound_checks_pass"] is True]
            coverage.append({
                "total_power_dbm": power, "xi": xi, **count_cases(group),
                "mean_bob_rate_passing_heldout": float(np.mean(rates)) if rates else None,
            })
    write_table(output / "coverage_by_power_xi.csv", coverage)
    by_split = {role: count_cases([row for row in cases if row["split"] == role]) for role in SPLITS}
    seconds = {
        "pso_and_label_bank": sum(s["seconds"] for s in searches),
        "validation": sum(s["seconds"] for s in selections),
        "heldout": sum(a["seconds"] for a in audits),
        "export": time.perf_counter() - started,
    }
    summary = {
        "step": "6a", "scope": plan["purpose"], "complete": True,
        "layouts_planned": len(plan["layouts"]),
        "layouts_eligible": sum(s["status"] == "eligible" for s in searches),
        "layout_status_counts": {status: sum(s["status"] == status for s in searches)
                                 for status in ("eligible", "invalid_geometry", "bob_not_strongest")},
        "total_power_dbm": plan["experiment"]["total_power_dbm"],
        "cases": count_cases(cases), "splits": by_split,
        "pso_evaluations": sum(s["pso_evaluations"] for s in searches), "seconds": seconds,
        "graph_exported": graph_exported,
        "all_splits_have_labels": all(by_split[role]["labeled"] > 0 for role in SPLITS),
        "heldout_used_for_label_filter": False,
        "graph_format": FORMAT,
    }
    save_json(output / "summary.json", summary)
    paths = sorted(path for path in output.rglob("*")
                   if path.is_file() and path.name != "run.lock" and path.suffix != ".tmp")
    save_json(output / "completion.json", {
        "sha256": {path.relative_to(output).as_posix(): sha256(path) for path in paths},
    })
    return summary


def check_run(output):
    output = Path(output)
    completion = read_json(output / "completion.json")
    for name, digest in completion["sha256"].items():
        if not (output / name).is_file() or sha256(output / name) != digest:
            raise ValueError(f"Step 6a result changed: {name}")
    plan = read_json(output / "plan.json")
    frozen = output / "selections_frozen.json"
    if read_json(frozen) != frozen_manifest(output, plan):
        raise ValueError("Frozen selections do not match the committed searches")
    cases = []
    for entry in plan["layouts"]:
        audit = read_committed(layout_folder(output, entry) / AUDIT)
        if audit is None or audit["frozen_sha256"] != sha256(frozen):
            raise ValueError("Missing audit or audit belongs to a different selection")
        cases.extend(audit["cases"])
    check_cases(plan, cases)
    summary = read_json(output / "summary.json")
    if summary["cases"] != count_cases(cases):
        raise ValueError("Summary counts differ from the committed results")
    if summary["graph_exported"]:
        check_graphs(output / "graphs", plan, cases)
    return summary


def generate(plan, output, resume=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "run.lock"
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise ValueError(f"Output is locked by a step 6a run: {lock}") from error
    try:
        with handle:
            handle.write(f"{os.getpid()}\n")
        plan_path = output / "plan.json"
        if any(path != lock for path in output.iterdir()):
            if not resume or not plan_path.exists() or read_json(plan_path) != plan:
                raise ValueError("Nonempty or changed run. Use --resume with the same plan, or a new output.")
            if (output / "completion.json").exists():
                return check_run(output)
        else:
            save_json(plan_path, plan)

        frozen_path = output / "selections_frozen.json"
        already_frozen = frozen_path.exists()
        if already_frozen and read_json(frozen_path) != frozen_manifest(output, plan):
            raise ValueError("Frozen selection changed. Audit cannot continue.")
        searches, selections = [], []
        for entry in plan["layouts"]:
            folder = layout_folder(output, entry)
            search = read_committed(folder / SEARCH)
            if search is None:
                print(f"Layout {entry['layout_id']} ({entry['split']}): PSO label search", flush=True)
                search = search_layout(plan, entry)
                commit(folder / SEARCH, search)
            search_hash = sha256(folder / SEARCH)
            selection = read_committed(folder / SELECTION)
            if selection is None:
                selection = select_layout(plan, entry, search, search_hash)
                commit(folder / SELECTION, selection)
            if selection["search_sha256"] != search_hash:
                raise ValueError(f"Validation selection belongs to a different search: {folder}")
            selected = sum(s["proposal_id"] is not None for p in selection["pools"] for s in p["selection"])
            print(f"Layout {entry['layout_id']}: {search['status']}, {selected} labels selected", flush=True)
            searches.append(search)
            selections.append(selection)
        if not already_frozen:
            save_json(frozen_path, frozen_manifest(output, plan))
        frozen_hash = sha256(frozen_path)
        print("All selections frozen. Checking on the independent fading bank.", flush=True)

        audits = []
        for entry, search, selection in zip(plan["layouts"], searches, selections):
            path = layout_folder(output, entry) / AUDIT
            audit = read_committed(path)
            if audit is None:
                audit = audit_layout(plan, entry, search, selection, frozen_hash)
                commit(path, audit)
            if audit["frozen_sha256"] != frozen_hash:
                raise ValueError(f"Audit belongs to a different frozen selection: {path}")
            counts = count_cases(audit["cases"])
            print(f"Layout {entry['layout_id']}: audit passed {counts['heldout_pass']}/{counts['labeled']} labels", flush=True)
            audits.append(audit)
        return summarize(output, plan, searches, selections, audits)
    finally:
        lock.unlink()
