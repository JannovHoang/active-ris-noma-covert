"""Generate a shared PSO pool and filter it at several DEP slack values."""
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
import platform

from filelock import FileLock
import numpy as np
import scipy
import torch

from .benchmark_pso import settings_for, validate_config_5c
from .budget_benchmark import CODE_FILES as PRIOR_CODE, check_budget, new_layout, record_path, run_phase
from .channels import build_bank, public_order
from .cross_xi import target_checks
from .expanded_benchmark import select_models, source_hashes
from .expanded_data import save_json, write_table
from .expanded_learning import validate_source
from .expanded_scenarios import ROOT, TRACKED, physical_config, reproduce_layout
from .graph_data import original_allocation, statistical_graph
from .learning_data import SPLITS, read_json, sha256
from .learning_train import configure_runtime, validate_training_config
from .objective import CachedObjective
from .pso import PSOSettings, optimize
from .validation import assess_proposal


STREAMS = {"label_search": 28, "label_pso": 29, "label_screen": 30, "model_validation": 31,
           "test_search": 32, "test_pso": 33, "warm_jitter": 34, "test_audit": 35, "bootstrap": 36}
LABEL_CODE = tuple(dict.fromkeys((*PRIOR_CODE, *TRACKED, "run_step6e.py", "ris_noma/dep_label_experiment.py")))


def inspect_experiment(dataset, training, benchmark, config, workers):
    expected = {"splits", "search_margins", "label_seed", "label_sweeps", "dep_slacks", "training_epochs",
                "training_patience", "training_batch_size", "bootstrap_repetitions"}
    if set(config) != expected or type(workers) is not int or workers < 1:
        raise ValueError("Invalid 6e configuration or worker count")
    _, schema = validate_source(dataset)
    check_budget(benchmark)
    prior = read_json(benchmark / "plan.json")
    hashes = source_hashes(dataset, training)
    if hashes != prior["sources_sha256"]:
        raise ValueError("Step 6d must have used the same 6a and 6b inputs")
    source = read_json(dataset / "plan.json")
    c = validate_config_5c(prior["search_config"])
    configure_runtime(0, c["threads"], "cpu")
    selected, _ = select_models(dataset, training, hashes, source["base_config"]["bounds"], schema)
    if prior["selected_checkpoint"] != selected["gnn"]:
        raise ValueError("The recorded 6d checkpoint differs from 6b")
    splits = config["splits"]
    if set(splits) != set(SPLITS) or any(not isinstance(splits[role], list) or not splits[role] for role in SPLITS):
        raise ValueError("Declare new layout IDs for train, validation and test")
    ids = [lid for role in SPLITS for lid in splits[role]]
    seen = {entry["layout_id"] for entry in source["layouts"]} | set(prior["layout_ids"])
    if (any(type(i) is not int or not 200 <= i <= 9999 for i in ids)
            or len(set(ids)) != len(ids) or seen.intersection(ids)):
        raise ValueError("Use distinct layout IDs outside 6a and 6d")
    for key in ("label_seed", "label_sweeps", "training_epochs", "training_patience",
                "training_batch_size", "bootstrap_repetitions"):
        minimum = 0 if key == "label_seed" else 1000 if key == "bootstrap_repetitions" else 1
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"Invalid {key}")
    powers, xis = source["experiment"]["total_power_dbm"], source["base_config"]["requirements"]["xi"]
    for key in ("search_margins", "dep_slacks"):
        values = config[key]
        if (not isinstance(values, list) or not values or any(type(v) not in (int, float) or not np.isfinite(v)
                or v < 0 or max(xis) + v >= 1. for v in values) or values != sorted(set(values))):
            raise ValueError(f"Invalid ordered DEP values: {key}")
    slacks, margins = config["dep_slacks"], config["search_margins"]
    if len(slacks) < 2 or slacks[0] != 0.:
        raise ValueError("Compare standard labels to at least one positive slack")
    train = read_json(training / "training_plan.json")["config"]
    train.update(models=["gnn"], epochs=config["training_epochs"], patience=config["training_patience"],
                 batch_size=config["training_batch_size"], notes="6e matched training on common layouts and inputs")
    validate_training_config(train)
    label_settings = replace(settings_for(c, "pso_full"), sweeps=config["label_sweeps"])
    roles = {lid: role for role in SPLITS for lid in splits[role]}
    layouts = [dict(new_layout(source, lid), split=roles[lid]) for lid in sorted(ids)]
    label_families = len(powers) * (len(splits["train"]) + len(splits["validation"]))
    samples = source["experiment"]["label_samples"]
    alpha = c["audit_alpha"]
    return {
        "step": "6e", **config, "layouts": layouts, "power_dbm": powers, "xi": xis, "feature_schema": schema,
        "sources_sha256": hashes, "source_6d_sha256": sha256(benchmark / "completion.json"),
        "historical_gnn": selected["gnn"], "label_settings": asdict(label_settings),
        "search_config": c, "training_config": train, "benchmark_budget": prior["primary_budget"],
        "search_samples_per_layout": samples, "label_screen_samples": source["experiment"]["validation_samples"],
        "audit_samples_per_layout": c["audit_samples"], "label_alpha_per_family": alpha / label_families,
        "validation_alpha_per_family": alpha / (len(splits["validation"]) * len(powers)),
        "test_alpha_per_family": alpha / (len(splits["test"]) * len(powers)),
        "label_evaluations_planned": label_families * len(xis) * len(margins) * label_settings.swarm_size * label_settings.sweeps,
        "workers": workers, "streams": STREAMS,
        "versions": {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version(),
                     "numpy": np.__version__, "scipy": scipy.__version__, "torch": str(torch.__version__)},
        "label_code_sha256": {name: sha256(ROOT / name) for name in LABEL_CODE},
    }, source


def rng(plan, physical, entry, stream, *indices):
    return np.random.default_rng(np.random.SeedSequence([physical["root_seed"], entry["layout_id"], plan["streams"][stream], *indices]))


def channel_context(plan, source, entry, stream, samples):
    base = physical_config(source, entry, plan["power_dbm"][0], samples)
    layout = reproduce_layout(base, entry)
    bank = build_bank(base, layout, rng(plan, base, entry, stream))
    return base, layout, bank


def label_search(entry, plan, source):
    record = {"layout_id": entry["layout_id"], "status": "invalid_geometry", "pools": [], "evaluations": 0}
    if not entry["geometry_valid"]:
        return record
    base, layout, bank = channel_context(plan, source, entry, "label_search", plan["search_samples_per_layout"])
    order = public_order(bank)
    if not CachedObjective(base, layout, bank, order).layout_eligible:
        return dict(record, status="bob_not_strongest")
    record.update(status="eligible", order=order.tolist())
    settings = PSOSettings(**plan["label_settings"])
    seed = plan["label_seed"]
    for power_idx, power in enumerate(plan["power_dbm"]):
        physical = dict(base, total_dbm=power)
        objective = CachedObjective(physical, layout, bank, order)
        graph = statistical_graph(physical, layout, bank, order, plan["xi"][0])
        proposals = []
        for (xi_idx, xi), (margin_idx, margin) in product(enumerate(plan["xi"]), enumerate(plan["search_margins"])):
            target = xi + margin
            result = optimize(lambda z: objective.evaluate(z, target), 3, settings,
                              rng(plan, base, entry, "label_pso", seed, power_idx, xi_idx, margin_idx))
            proposals.append({"proposal_id": f"p{len(proposals):03}", "xi": xi, "margin": margin, "seed": seed,
                              "gamma_target": target, "label_feasible": result.outcome.feasible,
                              "label_metrics": result.outcome.metrics, "evaluations": result.evaluations})
            record["evaluations"] += result.evaluations
        record["pools"].append({"total_power_dbm": power, "proposals": proposals,
                                "graph": {key: val.tolist() for key, val in graph.items()}})
    return record


def choose_labels(physical, order, proposals, assessments, xi, slacks):
    candidates = []
    for proposal in proposals:
        pid = proposal["proposal_id"]
        if not all(target_checks(physical, order, proposal, assessments[pid], xi).values()):
            continue
        lower = assessments[pid]["screen"]["dep_uniform_lower"]
        label = proposal["label_metrics"]
        y, _ = original_allocation(label["allocation_sic_order"], order, physical["bounds"])
        candidates.append({"proposal_id": pid, "source_xi": proposal["xi"], "gamma_target": proposal["gamma_target"],
                           "seed": proposal["seed"], "allocation": y.tolist(), "dep_lower": lower,
                           "bob_rate_label": label["bob_mean_rate_bit_s_hz"]})
    candidates.sort(key=lambda p: (-p["bob_rate_label"], p["gamma_target"] - xi, p["seed"], p["proposal_id"]))
    return [next((p for p in candidates if p["dep_lower"] >= xi + slack), None) for slack in slacks]


def label_screen(entry, plan, source, search, freeze):
    record = {"layout_id": entry["layout_id"], "split": entry["split"], "status": search["status"],
              "search_frozen_sha256": freeze, "cases": []}
    if search["status"] != "eligible":
        return record
    base, layout, bank = channel_context(plan, source, entry, "label_screen", plan["label_screen_samples"])
    order = np.asarray(search["order"], int)
    for pool in search["pools"]:
        power = pool["total_power_dbm"]
        physical = dict(base, total_dbm=power)
        proposals = [p for p in pool["proposals"] if p["label_feasible"]]
        assessments = {p["proposal_id"]: assess_proposal(physical, layout, bank, order, p, len(pool["proposals"]),
                                                         plan["label_alpha_per_family"], True) for p in proposals}
        for xi in plan["xi"]:
            labels = choose_labels(physical, order, proposals, assessments, xi, plan["dep_slacks"])
            record["cases"].append({"total_power_dbm": power, "xi": xi, "labels": labels})
    return record


def freeze_phase(output, phase, records, manifest):
    frozen = {"sha256": {record_path(Path(), phase, lid).as_posix(): sha256(record_path(output, phase, lid))
                         for lid in sorted(records)}}
    path = output / manifest
    if path.is_file():
        if read_json(path) != frozen:
            raise ValueError(f"Frozen {phase} records changed")
    else:
        save_json(path, frozen)
    return sha256(path)


def label_coverage(plan, screens):
    rows = []
    for role in ("train", "validation"):
        for xi in plan["xi"]:
            cases = [row for lid in sorted(screens) if screens[lid]["split"] == role
                     for row in screens[lid]["cases"] if row["xi"] == xi]
            matched = [row for row in cases if all(row["labels"])]
            for index, slack in enumerate(plan["dep_slacks"]):
                available = [row["labels"][index] for row in cases if row["labels"][index] is not None]
                delta_rate = [row["labels"][index]["bob_rate_label"] - row["labels"][0]["bob_rate_label"] for row in matched]
                rows.append({"split": role, "xi": xi, "dep_slack": slack, "eligible_cases": len(cases),
                             "labels": len(available), "matched_cases": len(matched),
                             "bob_rate_delta_on_matched_labels": float(np.mean(delta_rate)) if delta_rate else None})
    return rows


def prepare_labels(output, plan, source):
    entries = [e for e in plan["layouts"] if e["split"] in ("train", "validation")]
    workers = plan["workers"]
    searches = run_phase(output, "label_search", label_search, [(e, plan, source) for e in entries],
                         workers, (output / "label_search_frozen.json").is_file())
    freeze = freeze_phase(output, "label_search", searches, "label_search_frozen.json")
    screens = run_phase(output, "label_screen", label_screen, [(e, plan, source, searches[e["layout_id"]], freeze)
                        for e in entries], workers, (output / "labels_frozen.json").is_file())
    if any(record["search_frozen_sha256"] != freeze for record in screens.values()):
        raise ValueError("Label screening belongs to a different candidate pool")
    freeze_phase(output, "label_screen", screens, "labels_frozen.json")
    coverage = label_coverage(plan, screens)
    write_table(output / "label_coverage.csv", coverage)
    info = {"step": "6e", "labels_complete": True, "label_evaluations": sum(r["evaluations"] for r in searches.values()),
            "matched_labels": {role: sum(r["matched_cases"] for r in coverage if r["split"] == role and r["dep_slack"] == 0.)
                               for role in ("train", "validation")}, "coverage": coverage}
    save_json(output / "label_summary.json", info)
    return info, searches, screens


def run_experiment(args, plan, source):
    output = args.output.resolve()
    for original in (args.dataset, args.training, args.benchmark):
        path = original.resolve()
        if output == path or output in path.parents or path in output.parents:
            raise ValueError("Input and output directories must be separate")
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output.with_name(f".{output.name}.lock")), timeout=0):
        path = output / "plan.json"
        if output.exists() and any(output.iterdir()):
            if not args.resume or not path.is_file() or read_json(path) != plan:
                raise ValueError("Use a new output or --resume with the same inputs, plan and worker count")
        output.mkdir(parents=True, exist_ok=True)
        if (output / "completion.json").is_file():
            from .dep_label_learning import check_experiment
            return read_json(output / "label_summary.json") if args.command == "labels" else check_experiment(output)
        if not path.is_file():
            save_json(path, plan)
        info, searches, screens = prepare_labels(output, plan, source)
        if args.command == "labels":
            return info
        from .dep_label_learning import run_learning
        return run_learning(args, output, plan, source, info, searches, screens)
