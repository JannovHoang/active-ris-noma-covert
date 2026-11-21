"""Step 6f protocol: immutable 6e labels, two losses, two selection rules.

Training uses allocation labels only. Physical validation is forward-only,
on a finite checkpoint pool frozen before its independent bank is opened.
No physical audit or test result is backpropagated or used to repair outputs.
"""
from copy import deepcopy
from itertools import product
from pathlib import Path
import platform
import shutil

from filelock import FileLock
import numpy as np
import scipy
import torch

from .budget_benchmark import new_layout, record_path, run_phase
from .dep_label_experiment import LABEL_CODE, channel_context, freeze_phase
from .dep_label_learning import check_experiment as check_6e, graph_for, read_model, score_cases, write_immutable
from .expanded_data import save_json, write_table
from .expanded_learning import validate_source
from .expanded_scenarios import ROOT
from .learning_data import INPUT_KEYS, TARGET_KEY, file_fingerprints, physics_allocation, read_json, sha256
from .learning_train import configure_runtime, validate_training_config


STREAMS = {"model_validation": 37, "test_search": 38, "test_pso": 39,
           "warm_jitter": 40, "test_audit": 41, "bootstrap": 42}
GROUPS = ("uniform_mse", "uniform_physics", "weighted_mse", "weighted_physics")
CODE_FILES = tuple(dict.fromkeys((*LABEL_CODE, "ris_noma/dep_label_learning.py",
    "ris_noma/dep_label_benchmark.py", "run_step6f.py", "ris_noma/physics_training.py",
    "ris_noma/physics_training_fit.py", "ris_noma/physics_training_benchmark.py")))


def validate_config(config):
    fields = {"test_layout_ids", "epochs", "snapshot_epochs", "mb_error_weight", "bootstrap_repetitions"}
    if set(config) != fields:
        raise ValueError("Invalid step 6f config fields")
    ids, epochs, snapshots = config["test_layout_ids"], config["epochs"], config["snapshot_epochs"]
    if (not isinstance(ids, list) or len(ids) < 2 or any(type(i) is not int or not 500 <= i <= 9999 for i in ids)
            or ids != sorted(set(ids))):
        raise ValueError("Declare at least two distinct new test layout IDs in [500,9999]")
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if (not isinstance(snapshots, list) or not snapshots or snapshots != sorted(set(snapshots))
            or any(type(e) is not int or not 1 <= e <= epochs for e in snapshots) or snapshots[-1] != epochs):
        raise ValueError("Snapshot epochs must be distinct, increasing, and include the final epoch")
    if (type(config["mb_error_weight"]) not in (int, float)
            or not np.isfinite(config["mb_error_weight"]) or config["mb_error_weight"] <= 1):
        raise ValueError("mb_error_weight must be finite and greater than one")
    if type(config["bootstrap_repetitions"]) is not int or config["bootstrap_repetitions"] < 1000:
        raise ValueError("Declare at least 1000 layout-bootstrap repetitions")
    return deepcopy(config)


def source_fingerprints(dataset, previous):
    return {"dataset_completion": sha256(dataset / "completion.json"),
            "dataset_plan": sha256(dataset / "plan.json"),
            "previous_completion": sha256(previous / "completion.json"),
            "previous_inputs": {name: sha256(previous / name) for name in (
                "plan.json", "normalization.json", "matched_train.npz", "matched_validation.npz",
                "selection_frozen.json", "labels_frozen.json", "label_search_frozen.json")}}


def load_matched(previous, slack_index, splits):
    data = {}
    seen = set()
    for role in ("train", "validation"):
        with np.load(previous / f"matched_{role}.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        required = {*INPUT_KEYS, "labels", "layout_id", "xi", "total_power_dbm"}
        if not required.issubset(arrays) or not len(arrays["labels"]):
            raise ValueError("Incomplete frozen matched dataset")
        ids = set(arrays["layout_id"].tolist())
        if ids - set(splits[role]) or ids & seen:
            raise ValueError("Matched data has overlapping or incorrect layout splits")
        seen |= ids
        y = arrays["labels"][:, slack_index]
        if y.shape != (len(arrays["layout_id"]), 4) or not np.isfinite(y).all():
            raise ValueError("Invalid reference allocation dimensions")
        keys = list(zip(arrays["layout_id"], arrays["total_power_dbm"], arrays["xi"]))
        if len(set(keys)) != len(keys) or any(not np.isfinite(arrays[k]).all() for k in INPUT_KEYS):
            raise ValueError("Duplicate training cases or nonfinite graph inputs")
        arrays[TARGET_KEY] = y
        data[role] = arrays
    return data


def inspect_experiment(dataset, previous, config, workers):
    dataset, previous = Path(dataset).resolve(), Path(previous).resolve()
    config = validate_config(config)
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    _, schema = validate_source(dataset)
    check_6e(previous)
    old, source = read_json(previous / "plan.json"), read_json(dataset / "plan.json")
    expected = old["sources_sha256"]
    if (sha256(dataset / "completion.json") != expected["source_6a_completion"]
            or sha256(dataset / "plan.json") != expected["source_6a_plan"]
            or file_fingerprints(dataset / "graphs") != expected["source_6a_graphs"]
            or old["feature_schema"] != schema):
        raise ValueError("6e must have used this exact completed 6a dataset")
    # Reserve all older IDs below 500; also reject any IDs explicitly present in the inputs.
    prior_ids = {e["layout_id"] for e in source["layouts"] + old["layouts"]}
    if prior_ids & set(config["test_layout_ids"]):
        raise ValueError("New test layouts must be disjoint from every earlier input split")
    selection = read_json(previous / "selection_frozen.json")
    slack = selection["chosen_slack"]
    slack_index = old["dep_slacks"].index(slack)
    data = load_matched(previous, slack_index, old["splits"])
    for arrays in data.values():
        for y in arrays[TARGET_KEY]:
            physics_allocation(y, source["base_config"]["bounds"])
    training = deepcopy(old["training_config"])
    training.update(epochs=config["epochs"], notes="6f fixed epoch schedule; no early stopping; matched 6e labels")
    validate_training_config(training)
    seeds = training["seeds"]
    max_candidates = 2 * len(seeds) * (len(config["snapshot_epochs"]) + 1)
    test_layouts = [dict(new_layout(source, lid), split="test") for lid in config["test_layout_ids"]]
    validation = [e for e in old["layouts"] if e["split"] == "validation"]
    count_per_test = len(GROUPS) * len(seeds) + 3 * len(old["search_config"]["pso_seeds"])
    return {
        "step": "6f", **config, "workers": workers, "streams": STREAMS,
        "scope": "developmental K=2 study; new test layouts, same fixed Alice/RIS geometry distribution",
        "label_slack": slack, "label_slack_index": slack_index,
        "matched_cases": {r: len(d[TARGET_KEY]) for r, d in data.items()},
        "training_layout_ids": old["splits"]["train"], "validation_layout_ids": old["splits"]["validation"],
        "validation_layouts": validation, "test_layouts": test_layouts,
        "feature_schema": schema, "power_dbm": old["power_dbm"], "xi": old["xi"],
        "training_config": training, "search_config": old["search_config"],
        "benchmark_budget": old["benchmark_budget"], "search_samples_per_layout": old["search_samples_per_layout"],
        "audit_samples_per_layout": old["audit_samples_per_layout"],
        "validation_alpha_per_family": old["search_config"]["audit_alpha"] / (len(validation) * len(old["power_dbm"])),
        "test_alpha_per_family": old["search_config"]["audit_alpha"] / (len(test_layouts) * len(old["power_dbm"])),
        "loss_weights": {"uniform": [1., 1., 1., 1.], "weighted": [float(config["mb_error_weight"]), 1., 1., 1.]},
        "selection_groups": list(GROUPS), "max_validation_candidates": max_candidates,
        "validation_family_size": max_candidates * len(old["xi"]),
        "test_family_size": count_per_test * len(old["xi"]),
        "test_audit_rows_planned": len(test_layouts) * len(old["power_dbm"]) * len(old["xi"]) * count_per_test,
        "sources_sha256": source_fingerprints(dataset, previous),
        "normalization": read_json(previous / "normalization.json"),
        "code_sha256": {name: sha256(ROOT / name) for name in CODE_FILES},
        "versions": {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version(),
                     "numpy": np.__version__, "scipy": scipy.__version__, "torch": str(torch.__version__)},
        "selection_policy": "per fit: min allocation MSE or max validation physical passes, gated rate, lower MSE, epoch; "
                            "group: pooled validation passes, gated rate, declared group order; seed: passes, rate, seed",
        "test_used_for_training_or_selection": False,
    }, source


def validation_proposals(entry, plan, previous, refs, output, freeze):
    from .benchmark_pso import load_record, predict_start
    configure_runtime(0, plan["training_config"]["threads"], "cpu")
    search = load_record(record_path(previous, "label_search", entry["layout_id"]))
    if search is None or search["status"] != "eligible":
        raise ValueError("Need the frozen eligible 6e validation graph inputs")
    order = np.asarray(search["order"], int)
    models = {name: read_model(ref, output) for name, ref in refs.items()}
    predictions = []
    for pool, xi in product(search["pools"], plan["xi"]):
        graph = graph_for(pool, xi)
        for name, model in models.items():
            y, _ = predict_start(*model, graph, order, model[1]["bounds"])
            predictions.append({"layout_id": entry["layout_id"], "total_power_dbm": pool["total_power_dbm"],
                                "xi": xi, "candidate": name, "allocation": y.tolist()})
    return {"layout_id": entry["layout_id"], "fits_frozen_sha256": freeze, "order": order.tolist(), "rows": predictions}


def validation_audit(entry, plan, source, proposals, freeze):
    base, layout, bank = channel_context(plan, source, entry, "model_validation", plan["audit_samples_per_layout"])
    rows, unique = score_cases(base, layout, bank, np.asarray(proposals["order"], int), proposals["rows"],
                               plan["validation_family_size"], plan["validation_alpha_per_family"])
    return {"layout_id": entry["layout_id"], "validation_proposals_frozen_sha256": freeze, "rows": rows, "unique": unique}


def select_checkpoints(refs, rows, groups=GROUPS):
    """Only validation rows may enter this deterministic selector."""
    scores = {}
    for name, ref in refs.items():
        subset = [r for r in rows if r["candidate"] == name]
        if not subset:
            raise ValueError("Every candidate needs physical validation")
        scores[name] = {"candidate": name, "loss": ref["loss"], "seed": ref["seed"], "epoch": ref["epoch"],
                        "validation_mse": ref["validation_mse"], "runs": len(subset),
                        "passed": sum(r["audit_pass"] for r in subset),
                        "gated_rate": float(np.mean([r["gated_bob_rate"] for r in subset]))}
    selected = {}
    for loss, seed in sorted({(ref["loss"], ref["seed"]) for ref in refs.values()}):
        names = [name for name, ref in refs.items() if ref["loss"] == loss and ref["seed"] == seed]
        mse = [name for name in names if refs[name]["is_mse_best"]]
        if len(mse) != 1:
            raise ValueError("Exactly one allocation-MSE checkpoint is required per fit")
        physics = min(names, key=lambda n: (-scores[n]["passed"], -scores[n]["gated_rate"],
                                           refs[n]["validation_mse"], refs[n]["epoch"], n))
        for rule, name in (("mse", mse[0]), ("physics", physics)):
            selected[f"{loss}_{rule}_s{seed}"] = {**refs[name], "candidate": name, "group": f"{loss}_{rule}"}
    group_scores, representatives = [], {}
    for group in groups:
        names = [name for name, ref in selected.items() if ref["group"] == group]
        if not names:
            raise ValueError("Missing loss/selection ablation group")
        members = [scores[selected[name]["candidate"]] for name in names]
        group_scores.append({"group": group, "runs": sum(r["runs"] for r in members),
                             "passed": sum(r["passed"] for r in members),
                             "gated_rate": float(np.mean([r["gated_rate"] for r in members]))})
        representatives[group] = min(names, key=lambda n: (-scores[selected[n]["candidate"]]["passed"],
                                     -scores[selected[n]["candidate"]]["gated_rate"], selected[n]["seed"]))
    chosen = min(group_scores, key=lambda r: (-r["passed"], -r["gated_rate"], groups.index(r["group"])))
    return {"selected": selected, "representatives": representatives, "chosen_group": chosen["group"],
            "group_scores": group_scores, "candidate_scores": list(scores.values()), "test_used_for_selection": False}


def freeze_selection(output, previous, plan, source, refs):
    entries, workers = plan["validation_layouts"], plan["workers"]
    fits_hash = sha256(output / "fits_frozen.json")
    predictions = run_phase(output, "validation_proposals", validation_proposals,
        [(e, plan, previous, refs, output, fits_hash) for e in entries], workers,
        (output / "validation_proposals_frozen.json").is_file())
    if any(r["fits_frozen_sha256"] != fits_hash for r in predictions.values()):
        raise ValueError("Validation proposals belong to different checkpoints")
    frozen = freeze_phase(output, "validation_proposals", predictions, "validation_proposals_frozen.json")
    audits = run_phase(output, "validation_audit", validation_audit,
        [(e, plan, source, predictions[e["layout_id"]], frozen) for e in entries], workers,
        (output / "validation_frozen.json").is_file())
    if any(r["validation_proposals_frozen_sha256"] != frozen for r in audits.values()):
        raise ValueError("Validation audit belongs to different proposals")
    validation_hash = freeze_phase(output, "validation_audit", audits, "validation_frozen.json")
    rows = [r for lid in sorted(audits) for r in audits[lid]["rows"]]
    expected = set(product(audits, plan["power_dbm"], plan["xi"], refs))
    actual = {(r["layout_id"], r["total_power_dbm"], r["xi"], r["candidate"]) for r in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("Every validation case/checkpoint must be scored once")
    selection = select_checkpoints(refs, rows)
    selection.update(fits_frozen_sha256=fits_hash, validation_frozen_sha256=validation_hash)
    write_immutable(output / "selection_frozen.json", selection)
    write_table(output / "validation_candidates.csv", selection["candidate_scores"])
    write_table(output / "validation_groups.csv", selection["group_scores"])
    write_table(output / "validation_runs.csv", rows)
    print(f"Checkpoint/group selection frozen on validation: {selection['chosen_group']}", flush=True)
    return selection


def check_experiment(output):
    output = Path(output).resolve()
    hashes = read_json(output / "completion.json")["sha256"]
    required = {"plan.json", "fits_frozen.json", "validation_proposals_frozen.json", "validation_frozen.json",
                "selection_frozen.json", "test_plan.json", "test_search_frozen.json", "summary.json",
                "comparison_runs.csv", "comparison_by_method.csv", "paired_comparison.csv"}
    if not required.issubset(hashes):
        raise ValueError("Incomplete step 6f output")
    for name, digest in hashes.items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Step 6f file changed: {name}")
    plan, selection, summary = (read_json(output / n) for n in ("plan.json", "selection_frozen.json", "summary.json"))
    if (summary["step"] != "6f" or not summary["complete"] or summary["chosen_group"] != selection["chosen_group"]
            or summary["test_cases"] != summary["test_layouts"] * len(plan["power_dbm"]) * len(plan["xi"])
            or summary["test_audit_rows"] != summary["test_cases"] * (len(GROUPS) * len(plan["training_config"]["seeds"])
                                               + 3 * len(plan["search_config"]["pso_seeds"]))
            or selection["test_used_for_selection"] or plan["test_used_for_training_or_selection"]):
        raise ValueError("Step 6f counts or selection policy differ from its plan")
    for phase, manifest in (("validation_proposals", "validation_proposals_frozen.json"),
                             ("validation_audit", "validation_frozen.json"), ("test_search", "test_search_frozen.json")):
        frozen = read_json(output / manifest)["sha256"]
        if not frozen or any(hashes.get(n) != h for n, h in frozen.items()):
            raise ValueError(f"Invalid frozen {phase} ledger")
    return summary


def run_experiment(args, plan, source):
    from .physics_training_fit import fit_models
    from .physics_training_benchmark import run_test
    output, dataset, previous = (Path(p).resolve() for p in (args.output, args.dataset, args.previous))
    for original in (dataset, previous):
        if output == original or output in original.parents or original in output.parents:
            raise ValueError("Input/output directories must be separate and non-nested")
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output.with_name(f".{output.name}.lock")), timeout=0):
        path = output / "plan.json"
        if output.exists() and any(output.iterdir()):
            if not args.resume or not path.is_file() or read_json(path) != plan:
                raise ValueError("Use a new output or --resume with exactly the same plan and worker count")
        output.mkdir(exist_ok=True)
        if (output / "completion.json").is_file():
            return check_experiment(output)
        write_immutable(path, plan)
        write_immutable(output / "source_physics_plan.json", source)
        write_immutable(output / "normalization.json", plan["normalization"])
        data = load_matched(previous, plan["label_slack_index"], read_json(previous / "plan.json")["splits"])
        for role in data:
            path = output / f"matched_{role}.npz"
            if not path.exists():
                shutil.copyfile(previous / path.name, path)
            if sha256(path) != plan["sources_sha256"]["previous_inputs"][path.name]:
                raise ValueError("Copied matched dataset changed")
        refs = fit_models(output, plan, source["base_config"]["bounds"], data)
        selection = freeze_selection(output, previous, plan, source, refs)
        result = run_test(output, plan, source, selection)
        if source_fingerprints(dataset, previous) != plan["sources_sha256"]:
            raise ValueError("Source files changed during step 6f")
        save_json(output / "summary.json", result)
        save_json(output / "completion.json", {"sha256": {
            p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*"))
            if p.is_file() and p.suffix != ".tmp" and p != output / "completion.json"
            and (output / "figures") not in p.parents}})
        return check_experiment(output)
