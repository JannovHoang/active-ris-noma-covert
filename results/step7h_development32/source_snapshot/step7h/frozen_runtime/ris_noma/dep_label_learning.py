"""Fit matched GNNs and freeze the margin and checkpoints on validation."""
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .benchmark_pso import predict_start
from .budget_benchmark import run_phase
from .dep_label_experiment import channel_context, freeze_phase
from .expanded_data import save_json, write_table
from .expanded_scenarios import ROOT
from .graph_data import GLOBAL_FEATURES, fit_normalizer, transform_graphs
from .learning_data import TARGET_KEY, read_json, sha256
from .learning_physics import assess_allocation
from .learning_train import configure_runtime, load_checkpoint, train_one


def write_immutable(path, value):
    if path.is_file():
        if read_json(path) != value:
            raise ValueError(f"Frozen file differs: {path.name}")
    else:
        save_json(path, value)


def graph_for(pool, xi):
    graph = {key: np.asarray(value) for key, value in pool["graph"].items()}
    graph["global_raw"][GLOBAL_FEATURES.index("xi")] = xi
    return graph


def matched_data(output, searches, screens):
    data = {}
    for role in ("train", "validation"):
        rows = []
        for lid in sorted(screens):
            record = screens[lid]
            if record["split"] != role:
                continue
            pools = {p["total_power_dbm"]: p for p in searches[lid]["pools"]}
            for case in record["cases"]:
                if all(case["labels"]):
                    rows.append((lid, case, graph_for(pools[case["total_power_dbm"]], case["xi"])))
        if not rows:
            raise ValueError(f"Need cases with all label slacks in the {role} split")
        arrays = {key: np.stack([graph[key] for _, _, graph in rows]) for key in rows[0][2]}
        arrays.update(layout_id=np.array([lid for lid, _, _ in rows]),
                      xi=np.array([case["xi"] for _, case, _ in rows]),
                      total_power_dbm=np.array([case["total_power_dbm"] for _, case, _ in rows]),
                      labels=np.array([[label["allocation"] for label in case["labels"]] for _, case, _ in rows]))
        data[role] = arrays
    normalizer = fit_normalizer(data["train"]["node_raw"], data["train"]["global_raw"])
    write_immutable(output / "normalization.json", normalizer)
    fingerprints = {"normalization.json": sha256(output / "normalization.json")}
    for role, arrays in data.items():
        arrays["node"], arrays["global_features"] = transform_graphs(arrays["node_raw"], arrays["global_raw"], normalizer)
        path = output / f"matched_{role}.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as saved:
                if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
                    raise ValueError(f"Matched dataset differs: {path.name}")
        else:
            tmp = path.with_suffix(".tmp")
            with tmp.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            tmp.replace(path)
        fingerprints[path.name] = sha256(path)
    return data, normalizer, fingerprints


def read_model(reference, output):
    path = output / reference["path"]
    if sha256(path) != reference["sha256"]:
        raise ValueError(f"Checkpoint changed: {path}")
    return load_checkpoint(path)


def fit_models(output, plan, source, searches, screens):
    data, normalizer, fingerprints = matched_data(output, searches, screens)
    write_immutable(output / "learning_plan.json", {
        "step": "6e", "labels_frozen_sha256": sha256(output / "labels_frozen.json"),
        "config": plan["training_config"], "dataset_sha256": fingerprints,
        "matched_cases": {role: len(arrays["labels"]) for role, arrays in data.items()},
        "checkpoint_rule": "MSE within each fit, physical pass count and gated Bob rate across fits",
        "margin_rule": "highest validation pass count across all seeds, then gated Bob rate, then smaller positive slack",
        "code_sha256": sha256(ROOT / "ris_noma/dep_label_learning.py"),
    })
    config, bounds = plan["training_config"], source["base_config"]["bounds"]
    refs = {}
    fits = output / "fits"
    fits.mkdir(exist_ok=True)
    for (index, slack), seed in product(enumerate(plan["dep_slacks"]), config["seeds"]):
        name = f"d{index:02d}_s{seed}"
        folder = fits / name
        if not folder.exists():
            splits = {role: {**arrays, TARGET_KEY: arrays["labels"][:, index]} for role, arrays in data.items()}
            with TemporaryDirectory(prefix=".fit_", dir=output) as temp:
                stage = Path(temp) / "model"
                train_one("gnn", seed, config, bounds, splits["train"], splits["validation"], normalizer,
                          plan["feature_schema"], fingerprints, stage)
                stage.replace(folder)
        history = read_json(folder / "training_summary.json")
        reference = {"path": f"fits/{name}/best_model.pt", "sha256": history["checkpoint_sha256"],
                     "slack": slack, "slack_index": index, "seed": seed,
                     "epoch": history["best_epoch"], "validation_mse": history["validation"]["mse"]}
        _, checkpoint = read_model(reference, output)
        if (checkpoint["dataset_sha256"] != fingerprints or checkpoint["normalization"] != normalizer
                or checkpoint["seed"] != seed or checkpoint["config"] != config):
            raise ValueError(f"Checkpoint does not belong to this matched dataset: {name}")
        refs[name] = reference
        print(f"Fit {len(refs)} of {len(plan['dep_slacks']) * len(config['seeds'])} committed: {name}", flush=True)
    write_immutable(output / "fits_frozen.json", {"checkpoints": refs, "learning_plan_sha256": sha256(output / "learning_plan.json")})
    return refs


def score_cases(base, layout, bank, order, cases, family_size, alpha):
    cache, rows = {}, []
    for row in cases:
        power, xi = row["total_power_dbm"], row["xi"]
        key = power, xi, tuple(row["allocation"])
        if key not in cache:
            physical = dict(base, total_dbm=power)
            summary, screen = assess_allocation(physical, layout, bank, order, row["allocation"], xi, family_size, alpha)
            rate, passed = summary["bob_mean_rate_bit_s_hz"], screen["accepted"]
            qos = np.asarray(physical["requirements"]["qos_public"])[order]
            cache[key] = {
                "audit_pass": passed, "bob_rate": rate, "gated_bob_rate": rate if passed else 0.,
                "dep_gamma": summary["dep_min"], "dep_empirical": screen["dep_empirical_min"],
                "dep_lower": screen["dep_uniform_lower"], "bob_outage_upper": screen["bob_outage_upper"],
                "qos_lower_margin": float(np.min(np.asarray(screen["public_rate_lower_sic_order"]) - qos)),
                "failed_checks": [key for key, ok in screen["checks"].items() if not ok],
            }
        rows.append({**row, **cache[key]})
    return rows, len(cache)


def validate_layout(entry, plan, source, search, refs, output, freeze):
    order = np.asarray(search["order"], int)
    configure_runtime(0, plan["training_config"]["threads"], "cpu")
    models = {name: read_model(ref, output) for name, ref in refs.items()}
    bounds = source["base_config"]["bounds"]
    predictions = []
    for pool, xi in product(search["pools"], plan["xi"]):
        graph = graph_for(pool, xi)
        for name, model in models.items():
            y, _ = predict_start(*model, graph, order, bounds)
            predictions.append({"layout_id": entry["layout_id"], "total_power_dbm": pool["total_power_dbm"],
                                "xi": xi, "model": name, "allocation": y.tolist()})
    base, layout, bank = channel_context(plan, source, entry, "model_validation", plan["audit_samples_per_layout"])
    rows, _ = score_cases(base, layout, bank, order, predictions, len(refs) * len(plan["xi"]),
                          plan["validation_alpha_per_family"])
    return {"layout_id": entry["layout_id"], "fits_frozen_sha256": freeze, "rows": rows}


def freeze_selection(output, plan, source, searches, refs):
    entries = [e for e in plan["layouts"] if e["split"] == "validation" and searches[e["layout_id"]]["status"] == "eligible"]
    if not entries:
        raise ValueError("No eligible validation layouts")
    freeze = sha256(output / "fits_frozen.json")
    records = run_phase(output, "model_validation", validate_layout,
                        [(e, plan, source, searches[e["layout_id"]], refs, output, freeze) for e in entries],
                        plan["workers"], (output / "validation_frozen.json").is_file())
    if any(record["fits_frozen_sha256"] != freeze for record in records.values()):
        raise ValueError("Validation belongs to a different checkpoint set")
    validation_hash = freeze_phase(output, "model_validation", records, "validation_frozen.json")
    rows = [row for lid in sorted(records) for row in records[lid]["rows"]]
    expected = set(product(records, plan["power_dbm"], plan["xi"], refs))
    observed = {(r["layout_id"], r["total_power_dbm"], r["xi"], r["model"]) for r in rows}
    if expected != observed or len(rows) != len(expected):
        raise ValueError("Every validation checkpoint needs the same cases")
    scores = []
    for name, reference in refs.items():
        group = [r for r in rows if r["model"] == name]
        scores.append({"model": name, "slack": reference["slack"], "seed": reference["seed"], "runs": len(group),
                       "passed": sum(r["audit_pass"] for r in group),
                       "gated_rate": float(np.mean([r["gated_bob_rate"] for r in group]))})
    margin_scores = []
    for slack in plan["dep_slacks"]:
        group = [r for r in scores if r["slack"] == slack]
        margin_scores.append({"slack": slack, "runs": sum(r["runs"] for r in group),
                              "passed": sum(r["passed"] for r in group),
                              "gated_rate": float(np.mean([r["gated_rate"] for r in group]))})
    chosen = min(margin_scores[1:], key=lambda r: (-r["passed"], -r["gated_rate"], r["slack"]))["slack"]
    representatives = {}
    for name, slack in (("standard", 0.), ("robust", chosen)):
        representatives[name] = min((row for row in scores if row["slack"] == slack),
                                    key=lambda r: (-r["passed"], -r["gated_rate"], r["seed"]))["model"]
    selection = {"chosen_slack": chosen, "representatives": representatives,
                 "validation_frozen_sha256": validation_hash, "fits_frozen_sha256": freeze}
    write_immutable(output / "selection_frozen.json", selection)
    write_table(output / "validation_models.csv", scores)
    write_table(output / "validation_margins.csv", margin_scores)
    print(f"Validation selection frozen. Chosen positive slack: {chosen:g}", flush=True)
    return selection


def check_experiment(output):
    output = Path(output).resolve()
    hashes = read_json(output / "completion.json")["sha256"]
    required = {"plan.json", "labels_frozen.json", "label_summary.json", "label_coverage.csv", "learning_plan.json",
                "fits_frozen.json", "selection_frozen.json", "test_search_frozen.json", "summary.json",
                "comparison_runs.csv", "comparison_by_method.csv", "paired_comparison.csv"}
    if not required.issubset(hashes):
        raise ValueError("Incomplete step 6e output")
    for name, digest in hashes.items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Step 6e result changed: {name}")
    result, plan, selection = (read_json(output / name) for name in ("summary.json", "plan.json", "selection_frozen.json"))
    if (result["step"] != "6e" or not result["complete"] or result["chosen_slack"] != selection["chosen_slack"]
            or result["test_cases"] != result["test_layouts"] * len(plan["xi"]) * len(plan["power_dbm"])
            or result["test_audit_rows"] != result["test_cases"] * (1 + 2 * len(plan["training_config"]["seeds"])
                                                                  + 3 * len(plan["search_config"]["pso_seeds"]))):
        raise ValueError("Step 6e output counts differ from the frozen plan")
    return result


def run_learning(args, output, plan, source, info, searches, screens):
    refs = fit_models(output, plan, source, searches, screens)
    selection = freeze_selection(output, plan, source, searches, refs)
    from .dep_label_benchmark import run_test
    result = run_test(args, output, plan, source, info, refs, selection)
    save_json(output / "summary.json", result)
    save_json(output / "completion.json", {"sha256": {
        path.relative_to(output).as_posix(): sha256(path) for path in sorted(output.rglob("*"))
        if path.is_file() and path.suffix != ".tmp" and path != output / "completion.json"
        and not any(part.startswith(".") for part in path.relative_to(output).parts)
        and (output / "figures") not in path.parents
    }})
    return check_experiment(output)
