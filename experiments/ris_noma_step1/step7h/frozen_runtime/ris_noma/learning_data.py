"""Step 5b: validate an unchanged 5a export and expose only permitted inputs.

The audit ledger is for reporting, never for filtering or model features.
No source-study directory is needed to train on an exported graphs directory.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from .dataset import check_dataset
from .graph_data import SCHEMA_VERSION, NODE_FEATURES, GLOBAL_FEATURES, EDGE_FEATURES

SPLITS = ("train", "validation", "test")
INPUT_KEYS = ("node", "global_features", "edge_index", "edge_attr")
TARGET_KEY = "y_allocation_original_order"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_fingerprints(root):
    return {p.name: sha256(p) for p in sorted(Path(root).iterdir()) if p.is_file()}


def load_split(root, role):
    if role not in SPLITS:
        raise ValueError("Unknown split")
    with np.load(Path(root) / (role + ".npz"), allow_pickle=False) as archive:
        return {k: archive[k].copy() for k in archive.files}


def inspect_export(root):
    root = Path(root)
    status = check_dataset(root)
    manifest, schema = read_json(root / "dataset_manifest.json"), read_json(root / "feature_schema.json")
    if manifest["schema"] != SCHEMA_VERSION or schema["version"] != SCHEMA_VERSION:
        raise ValueError("Step 5b requires statistical_sic_graph_k2_v1")
    for field, expected in [("node_features", NODE_FEATURES), ("global_features", GLOBAL_FEATURES),
                            ("edge_features", EDGE_FEATURES),
                            ("node_order", ["Bob", "public_original_0", "public_original_1"]),
                            ("target_order", ["mb", "public_original_0", "public_original_1", "s"])]:
        if schema[field] != expected:
            raise ValueError(f"Unexpected schema order: {field}")
    records = [json.loads(line) for line in (root / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    ledger = {(r["split"], r["layout_id"], r["xi"]): r for r in records}
    if len(ledger) != len(records) or any(r.get("heldout_used_for_label_filter") is not False for r in records):
        raise ValueError("Duplicate ledger key or heldout-based label filtering")
    for role in SPLITS:
        d = load_split(root, role)
        seen = set()
        for k, (lid, xi) in enumerate(zip(d["layout_id"], d["xi"])):
            key = (role, int(lid), float(xi))
            if key in seen or key not in ledger:
                raise ValueError("Missing or duplicated graph ledger entry")
            seen.add(key)
            record = ledger[key]
            if not record["label_available"] or not np.allclose(record["allocation_original_order"], d[TARGET_KEY][k], rtol=0, atol=1e-12):
                raise ValueError("Graph label differs from ledger")
            order = d["sic_order_original_ids"][k]
            if not np.array_equal(record["sic_order_original_ids"], order):
                raise ValueError("Graph SIC order differs from ledger")
            ranks = np.empty(3, int); ranks[0] = 3; ranks[order + 1] = [1, 2]
            pairs = [(src, dst) for src in range(3) for dst in range(3) if src != dst]
            expected_edges = np.asarray(pairs, int).T
            expected_attr = np.asarray([[src != 0 and ranks[src] < ranks[dst], src == 0 or ranks[src] > ranks[dst]] for src, dst in pairs], float)
            if not np.array_equal(d["edge_index"][k], expected_edges) or not np.array_equal(d["edge_attr"][k], expected_attr):
                raise ValueError("Unexpected SIC graph topology/attributes")
        expected_keys = {key for key, r in ledger.items() if key[0] == role and r["label_available"]}
        if seen != expected_keys:
            raise ValueError("Labeled ledger rows and split differ")
    reviews = []
    for r in records:
        if r.get("heldout_bound_checks_pass") is False:
            a = r["audit_summary"]
            reviews.append({"layout_id": r["layout_id"], "xi": r["xi"], "split": r["split"],
                            "proposal_id": r["proposal_id"], "retained": True, **a,
                            "known_dep_bound_shortfall": max(0., r["xi"] - a["dep_lower_heldout"]),
                            "interpretation": "Recorded empirical point passes; inspect reproduced individual checks for all bound failures."})
    return {**status, "input_files_sha256": file_fingerprints(root), "audit_reviews": reviews,
            "model_input_keys": list(INPUT_KEYS), "labels_filtered_using_audit": False,
            "normalizer_fitted_on": "train only", "test_scope": "developmental pilot, not final independent GNN test"}


def physics_allocation(values, bounds):
    """Only correct float32 rounding; do not repair DEP/QoS/outage violations."""
    y = np.asarray(values, dtype=np.float64).copy()
    if y.shape != (4,) or not np.all(np.isfinite(y)):
        raise ValueError("Need four finite allocation values")
    for i, name in [(0, "mb"), (3, "s")]:
        lo, hi = bounds[name]
        if not lo - 1e-6 <= y[i] <= hi + 1e-6:
            raise ValueError("Prediction outside declared power bounds")
        y[i] = np.clip(y[i], lo, hi)
    if np.any(y[1:3] < -1e-6) or abs(y[:3].sum() - 1.) > 1e-6 or y[1:3].sum() <= 0:
        raise ValueError("Prediction outside the power simplex")
    weights = np.maximum(y[1:3], 0.)
    y[1:3] = (1. - y[0]) * weights / weights.sum()
    return y
