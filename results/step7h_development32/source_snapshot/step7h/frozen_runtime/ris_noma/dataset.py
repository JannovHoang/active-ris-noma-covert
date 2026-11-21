"""Read-only export of step-4b references into grouped statistical graphs.

Labels are selected on validation, regardless of their later heldout outcome.
Heldout flags remain in the ledger for coverage and audit reporting.
"""
from copy import deepcopy
from pathlib import Path
import json
import time
import numpy as np
import run_step3b as step3b
import run_step4a as step4a
import run_step4b as step4b
from .config import validate_config
from .geometry import sample_layout
from .channels import build_bank, public_order
from .random_streams import make_rng
from .objective import CachedObjective
from .graph_data import (SCHEMA_VERSION, NODE_FEATURES, GLOBAL_FEATURES, EDGE_FEATURES,
                         statistical_graph, original_allocation, fit_normalizer, transform_graphs)

ROOT = Path(__file__).resolve().parents[1]
NEW_CODE = ["ris_noma/graph_data.py", "ris_noma/dataset.py", "run_step5a.py"]
SPLITS = ["train", "validation", "test"]


def validate_partition(config, layout_ids):
    if config.get("purpose") not in ["development", "pilot"] or set(config["splits"]) != set(SPLITS):
        raise ValueError("Need development/pilot purpose and train/validation/test splits")
    pairs = [(lid, role) for role in SPLITS for lid in config["splits"][role]]
    ids = [p[0] for p in pairs]
    if not ids or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Layout IDs must be distinct nonnegative integers across splits")
    if set(ids) != set(layout_ids):
        raise ValueError("Partition must cover exactly all source layouts")
    if not config["splits"]["train"]:
        raise ValueError("Train split must contain layouts")
    if any(i >= 10000 for i in ids):
        raise ValueError("Layout IDs >= 10000 are reserved for a separately planned final test")
    if any(lid <= 5 and role != "train" for lid, role in pairs):
        raise ValueError("Previously examined layout IDs 0..5 must remain development/train")
    if config["purpose"] == "development" and any(config["splits"][r] for r in SPLITS[1:]):
        raise ValueError("Development export puts all layouts in train; it is not a generalization benchmark")
    return dict(pairs)


def source_fingerprints(source, label_source=None):
    source = Path(source).resolve()
    plan = step4a.read_json(source / "study_plan.json")
    status = step4a.read_json(source / "study_status.json")
    if plan["step"] != "4b" or status.get("step") != "4b" or not status["study_complete"] or status["heldout_used_for_selection"]:
        raise ValueError("Need a completed step-4b study with heldout excluded from selection")
    label_source = Path(label_source if label_source is not None else plan["source_study"]).resolve()
    old_plan, _, old_manifest = step4b.source_plan(label_source, plan["fresh_heldout_seed"])
    if old_manifest != plan["source_sha256"]:
        raise ValueError("Original step-4a source differs from the one used by step 4b")
    manifest = {"4a/" + k: v for k, v in old_manifest.items()}
    for name in ["study_plan.json", "study_status.json"]:
        manifest["4b/" + name] = step4a.sha(source / name)
    for lid in plan["study"]["layout_ids"]:
        folder = source / f"layout_{lid:06d}"
        if not step4a.completed_job(folder):
            raise ValueError(f"Incomplete step-4b layout {lid}")
        for name in ["completion.json", *step4a.RECEIPT_FILES]:
            path = folder / name
            if path.exists():
                manifest[f"4b/{folder.name}/{name}"] = step4a.sha(path)
    return source, label_source, plan, old_plan, manifest


def pack_graphs(items, normalizer):
    n = len(items)
    def array(key, shape, dtype=float):
        return np.stack([i[key] for i in items]).astype(dtype) if n else np.empty((0, *shape), dtype=dtype)
    nodes = array("node_raw", (3, len(NODE_FEATURES)))
    global_features = array("global_raw", (len(GLOBAL_FEATURES),))
    normalized_node, normalized_global = transform_graphs(nodes, global_features, normalizer)
    return {"node_raw": nodes, "global_raw": global_features,
            "node": normalized_node, "global_features": normalized_global,
            "edge_index": array("edge_index", (2, 6), np.int64), "edge_attr": array("edge_attr", (6, 2)),
            "y_allocation_original_order": array("y", (4,)), "y_pso_sic": array("z", (3,)),
            "sic_order_original_ids": array("order", (2,), np.int64),
            "layout_id": np.asarray([i["layout_id"] for i in items], np.int64),
            "xi": np.asarray([i["xi"] for i in items], float),
            "bob_reference_rate_label": np.asarray([i["rate_label"] for i in items], float)}


def check_dataset(output):
    """Validate hashes, partitioning, shapes, mapping and normalization on disk."""
    output = Path(output)
    manifest = step4a.read_json(output / "dataset_manifest.json")
    for name, digest in manifest["output_sha256"].items():
        if step4a.sha(output / name) != digest:
            raise ValueError(f"Dataset output changed: {name}")
    normalizer = step4a.read_json(output / "normalization.json")
    ids_by_split, total = {}, 0
    bounds = manifest["bounds"]
    for role in SPLITS:
        with np.load(output / f"{role}.npz", allow_pickle=False) as data:
            n = len(data["layout_id"])
            if n != manifest["counts"][role]["labeled"]:
                raise ValueError("Dataset sample count mismatch")
            expected = {"node_raw": (n, 3, len(NODE_FEATURES)), "node": (n, 3, len(NODE_FEATURES)),
                        "global_raw": (n, len(GLOBAL_FEATURES)), "global_features": (n, len(GLOBAL_FEATURES)),
                        "edge_index": (n, 2, 6), "edge_attr": (n, 6, 2),
                        "y_allocation_original_order": (n, 4), "y_pso_sic": (n, 3),
                        "sic_order_original_ids": (n, 2)}
            for key, shape in expected.items():
                if data[key].shape != shape or not np.all(np.isfinite(data[key])):
                    raise ValueError(f"Invalid shape/values: {role}/{key}")
            ids_by_split[role] = set(data["layout_id"].tolist())
            if not ids_by_split[role] <= set(manifest["partition"]["splits"][role]):
                raise ValueError("Samples are in the wrong layout split")
            if not np.allclose(data["global_raw"][:, 0], data["xi"], rtol=0, atol=0):
                raise ValueError("Xi feature does not match sample key")
            tn, tg = transform_graphs(data["node_raw"], data["global_raw"], normalizer)
            if not np.allclose(tn, data["node"], rtol=1e-12, atol=1e-12) or not np.allclose(tg, data["global_features"], rtol=1e-12, atol=1e-12):
                raise ValueError("Saved feature normalization does not reconcile")
            for k in range(n):
                y, z, order = data["y_allocation_original_order"][k], data["y_pso_sic"][k], data["sic_order_original_ids"][k]
                expected_y, expected_z = original_allocation({"mb": y[0], "public": y[1:3][order].tolist(), "s": y[3]}, order, bounds)
                if not np.allclose(y, expected_y, rtol=0, atol=1e-12) or not np.allclose(z, expected_z, rtol=0, atol=1e-12):
                    raise ValueError("SIC target mapping mismatch")
            total += n
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            if ids_by_split[a] & ids_by_split[b]:
                raise ValueError("Layout leakage between splits")
    with np.load(output / "train.npz", allow_pickle=False) as train:
        fitted = fit_normalizer(train["node_raw"], train["global_raw"])
    for key in ["node", "global"]:
        for field in ["mean", "scale"]:
            if not np.allclose(fitted[key][field], normalizer[key][field], rtol=1e-12, atol=1e-12):
                raise ValueError("Normalizer was not fitted on train only")
    return {"schema": manifest["schema"], "checks": "passed", "labeled_graphs": total,
            "counts": manifest["counts"], "purpose": manifest["partition"]["purpose"],
            "ready_for_training_pilot": bool(manifest["partition"]["purpose"] == "pilot" and manifest["counts"]["train"]["labeled"] > 0 and manifest["counts"]["validation"]["labeled"] > 0),
            "final_gnn_test": False, "trained_model": False}


def export_dataset(source, output, partition=None, label_source=None, resume=False):
    started = time.perf_counter()
    source, label_source, plan, old_plan, fingerprints = source_fingerprints(source, label_source)
    output = Path(output).resolve()
    for src in [source, label_source]:
        if output == src or src in output.parents or output in src.parents:
            raise ValueError("Dataset output and source studies must be separate, non-nested directories")
    ids = plan["study"]["layout_ids"]
    if partition is None:
        partition = {"purpose": "development", "splits": {"train": ids, "validation": [], "test": []}}
    roles = validate_partition(partition, ids)
    export_plan = {"schema": SCHEMA_VERSION, "source": str(source), "label_source": str(label_source),
                   "partition": partition, "source_sha256": fingerprints,
                   "code_sha256": {name: step4a.sha(ROOT / name) for name in NEW_CODE},
                   "versions": plan["versions"], "source_stage": "4b"}
    if output.exists() and any(output.iterdir()):
        if not resume or not (output / "export_plan.json").exists() or step4a.read_json(output / "export_plan.json") != export_plan:
            raise ValueError("Nonempty dataset output or changed source/code/partition; use a new output")
        if (output / "dataset_manifest.json").exists():
            return check_dataset(output)
        step4a.archive_partial(output)
    output.mkdir(parents=True, exist_ok=True)
    step3b.write_json(output / "export_plan.json", export_plan)
    items = {r: [] for r in SPLITS}
    ledger = []
    feature_seconds = 0.
    for lid in ids:
        folder, original = source / f"layout_{lid:06d}", label_source / f"layout_{lid:06d}"
        status = step4a.read_json(folder / "status.json")
        if status["status"] == "layout_rejected_by_label_bob_mean_gain_rule":
            for xi in old_plan["base_config"]["requirements"]["xi"]:
                ledger.append({"layout_id": lid, "xi": xi, "split": roles[lid], "label_available": False,
                               "status": status["status"], "heldout_used_for_label_filter": False})
            continue
        c = validate_config(step4a.read_json(original / "config_label_used.json"))
        info = step4a.read_json(folder / "layout.json")
        order = np.asarray(info["sic_order_original_ids"], int)
        rows = step4a.read_json(folder / "selected_by_xi.json")
        frozen = {r["xi"]: r["proposal_id"] for r in step4a.read_json(folder / "selection_frozen.json")["selection"]}
        if {r["xi"] for r in rows} != set(c["requirements"]["xi"]):
            raise ValueError("Source xi rows do not match configuration")
        proposals = {p["proposal_id"]: p for p in step4a.read_json(original / "proposals_label.json")}
        feature_start = time.perf_counter()
        layout = sample_layout(c, make_rng(c["root_seed"], lid, "geometry"))
        if not np.allclose(layout.public_xy, info["public_xy_original_order"], rtol=0, atol=1e-12):
            raise ValueError("Source layout cannot be reproduced")
        bank = build_bank(c, layout, make_rng(c["root_seed"], lid, "label"))
        if not np.array_equal(public_order(bank), order):
            raise ValueError("Regenerated label-bank SIC order differs from source")
        # One graph per layout; xi is the only feature changing across its targets.
        base_graph = statistical_graph(c, layout, bank, order, c["requirements"]["xi"][0])
        selected = [r for r in rows if r["proposal_id"] is not None]
        if selected:
            p = proposals[selected[0]["proposal_id"]]
            z = original_allocation(p["label_metrics"]["allocation_sic_order"], order, c["bounds"])[1]
            result = CachedObjective(c, layout, bank, order).evaluate(z, p["xi"])
            for key in ["bob_mean_rate_bit_s_hz", "dep_min"]:
                if not np.isclose(result.metrics[key], p["label_metrics"][key], rtol=1e-9, atol=1e-10):
                    raise ValueError("Regenerated label bank does not reconcile with saved reference metrics")
        del bank
        feature_seconds += time.perf_counter() - feature_start
        for row in rows:
            xi, pid = row["xi"], row["proposal_id"]
            if frozen.get(xi) != pid or row["heldout_used_for_selection"]:
                raise ValueError("Selection was changed after freezing or used heldout")
            record = {"layout_id": lid, "root_seed": c["root_seed"], "xi": xi, "split": roles[lid],
                      "proposal_id": pid, "label_available": pid is not None, "status": row["status"],
                      "heldout_bound_checks_pass": row["heldout_bound_checks_pass"],
                      "heldout_used_for_label_filter": False, "sic_order_original_ids": order.tolist()}
            if pid is not None:
                p = proposals[pid]
                y, z = original_allocation(p["label_metrics"]["allocation_sic_order"], order, c["bounds"])
                expected = np.r_[row["mb"], np.array([row["m1"], row["m2"]])[np.argsort(order)], row["s"]]
                if not np.allclose(y, expected, rtol=0, atol=1e-12):
                    raise ValueError("Selected row and original proposal allocations differ")
                graph = {k: v.copy() for k, v in base_graph.items()}
                graph["global_raw"][0] = xi
                graph.update(y=y, z=z, order=order.copy(), layout_id=lid, xi=xi, rate_label=row["bob_rate_label"])
                # Deliberately keep audit-review labels; no post-hoc heldout filtering.
                items[roles[lid]].append(graph)
                record.update(allocation_original_order=y.tolist(), allocation_pso_sic=z.tolist(),
                              source_xi=row["source_xi"], source_margin=row["source_margin"],
                              effective_margin=row["margin"], bob_reference_rate_label=row["bob_rate_label"],
                              label_metrics=p["label_metrics"],
                              audit_summary={k: row[k] for k in ["dep_empirical_validation", "dep_lower_validation",
                                  "bob_rate_heldout", "dep_empirical_heldout", "dep_lower_heldout",
                                  "heldout_point_checks_pass", "heldout_bound_checks_pass"]})
            ledger.append(record)
        print(f"Graph export layout={lid}: {len(selected)}/{len(rows)} validation-selected labels", flush=True)
    with (output / "records.jsonl").open("w", encoding="utf-8") as f:
        for record in ledger:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    if not items["train"]:
        raise ValueError("No validation-selected training labels; see records.jsonl; no synthetic labels were generated")
    normalizer = fit_normalizer(np.stack([i["node_raw"] for i in items["train"]]),
                                np.stack([i["global_raw"] for i in items["train"]]))
    normalizer["training_layout_ids"] = sorted({i["layout_id"] for i in items["train"]})
    step3b.write_json(output / "normalization.json", normalizer)
    for role in SPLITS:
        np.savez_compressed(output / f"{role}.npz", **pack_graphs(items[role], normalizer))
    schema = {"version": SCHEMA_VERSION, "node_features": NODE_FEATURES, "global_features": GLOBAL_FEATURES,
              "edge_features": EDGE_FEATURES, "node_order": ["Bob", "public_original_0", "public_original_1"],
              "target_order": ["mb", "public_original_0", "public_original_1", "s"],
              "pso_target_order": ["z_mb", "z_public_sic_rank_1_share", "z_s"],
              "gain_source": "rho-independent tilde gains from the regenerated label bank, before any allocation",
              "edge_meaning": "source stream -> decoding receiver; computational relation, not physical link",
              "forbidden_model_inputs": ["layout_id", "seed", "proposal_id", "delta", "reference_rate", "allocation", "audit_result"]}
    step3b.write_json(output / "feature_schema.json", schema)
    counts = {r: {"planned": sum(x["split"] == r for x in ledger), "labeled": len(items[r]),
                  "missing": sum(x["split"] == r and not x["label_available"] for x in ledger),
                  "heldout_review_labels_retained": sum(x["split"] == r and x["label_available"] and x["heldout_bound_checks_pass"] is False for x in ledger)} for r in SPLITS}
    output_names = ["feature_schema.json", "normalization.json", "records.jsonl", "export_plan.json", *[r + ".npz" for r in SPLITS]]
    base_config = deepcopy(old_plan["base_config"])
    base_config["n_fading"] = plan["policy_per_layout"]["label_samples"]
    manifest = {"schema": SCHEMA_VERSION, "partition": partition, "counts": counts,
                "bounds": old_plan["base_config"]["bounds"], "label_bank_samples_per_layout": plan["policy_per_layout"]["label_samples"],
                "base_physics_config": base_config, "label_generation_policy": plan["policy_per_layout"],
                "source_study_alpha": plan["study"]["study_alpha"],
                "source_root_seed": old_plan["base_config"]["root_seed"],
                "source_sha256": fingerprints, "output_sha256": {n: step4a.sha(output / n) for n in output_names},
                "feature_generation_seconds": feature_seconds, "export_seconds": time.perf_counter() - started,
                "label_filter": "validation selection only; all heldout outcomes retained in ledger",
                "scope": "development/pilot graphs; reference PSO solutions are not proven global optima; no GNN trained"}
    step3b.write_json(output / "dataset_manifest.json", manifest)
    result = check_dataset(output)
    step3b.write_json(output / "dataset_status.json", result)
    return result
