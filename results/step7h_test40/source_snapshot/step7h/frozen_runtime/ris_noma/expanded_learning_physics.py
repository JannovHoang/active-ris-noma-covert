"""Forward-check frozen 6b allocations using the full layout/power/DEP key."""
from pathlib import Path
import time

import numpy as np

from .channels import build_bank, public_order
from .expanded_data import make_bank
from .expanded_scenarios import physical_config
from .graph_data import original_allocation, statistical_graph
from .learning_data import TARGET_KEY, physics_allocation, sha256
from .learning_physics import assess_allocation
from .objective import CachedObjective


CASE_FIELDS = ("layout_id", "total_power_dbm", "xi")
PREDICTION_FIELDS = (*CASE_FIELDS, "sic_order_original_ids")
AUDIT_STREAM = 19
AUDIT_SPLITS = ("validation", "test")


def load_prediction(path, digest, data):
    """Require the same case keys, order, and references as the audited split."""
    if sha256(path) != digest:
        raise ValueError(f"Frozen prediction changed: {path.name}")
    with np.load(path, allow_pickle=False) as arrays:
        for field in PREDICTION_FIELDS:
            if not np.array_equal(arrays[field], data[field]):
                raise ValueError(f"Prediction identity mismatch in {path.name}: {field}")
        if not np.array_equal(arrays["reference"], data[TARGET_KEY]):
            raise ValueError(f"Prediction reference mismatch: {path.name}")
        prediction = arrays["prediction"]
        if prediction.shape != data[TARGET_KEY].shape or not np.all(np.isfinite(prediction)):
            raise ValueError(f"Invalid prediction values: {path.name}")
        return prediction


def verify_label_context(plan, entry, data, indices):
    """Regenerate the bank that defined the input features and SIC order."""
    c, layout, bank = make_bank(plan, entry, "label")
    order = public_order(bank)
    if not np.all(data["sic_order_original_ids"][indices] == order):
        raise ValueError(f"Source SIC order changed for layout {entry['layout_id']}")
    for power in np.unique(data["total_power_dbm"][indices]):
        group = indices[data["total_power_dbm"][indices] == power]
        physical = dict(c, total_dbm=float(power))
        graph = statistical_graph(physical, layout, bank, order, data["xi"][group[0]])
        global_features = np.repeat(graph["global_raw"][None, :], len(group), axis=0)
        global_features[:, 0] = data["xi"][group]
        if not np.allclose(global_features, data["global_raw"][group], rtol=1e-9, atol=1e-10):
            raise ValueError(f"Source global features changed for layout {entry['layout_id']}, {power} dBm")
        for field in ("node_raw", "edge_index", "edge_attr"):
            if not np.allclose(graph[field], data[field][group], rtol=1e-9, atol=1e-10):
                raise ValueError(f"Source graph differs for layout {entry['layout_id']}: {field}")
        objective = CachedObjective(physical, layout, bank, order)
        if not objective.layout_eligible:
            raise ValueError("Regenerated label bank violates the source layout rule")
        for i in group:
            outcome = objective.evaluate(data["y_pso_sic"][i], float(data["xi"][i]))
            rate = outcome.metrics["bob_mean_rate_bit_s_hz"]
            if not np.isclose(rate, data["bob_reference_rate_label"][i], rtol=1e-9, atol=1e-10):
                raise ValueError(f"Source PSO rate changed for layout {entry['layout_id']}, {power} dBm")
    return layout, order


def audit_predictions(source_plan, splits, names, prediction_dir, prediction_hashes, samples, seed, alpha):
    """Use one fresh bank per layout and paired fading across all allocations."""
    started = time.perf_counter()
    prediction_dir = Path(prediction_dir)
    layouts = {entry["layout_id"]: entry for entry in source_plan["layouts"]}
    n_families = sum(
        len(set(zip(splits[role]["layout_id"], splits[role]["total_power_dbm"])))
        for role in AUDIT_SPLITS
    )
    alpha_family = alpha / n_families
    rows = []
    for role in AUDIT_SPLITS:
        data = splits[role]
        predictions = {}
        for name in names:
            file = f"predictions_{name}_{role}.npz"
            predictions[name] = load_prediction(prediction_dir / file, prediction_hashes[file], data)
        predictions["pso_reference"] = data[TARGET_KEY]
        for lid in sorted(set(data["layout_id"].tolist())):
            entry = layouts[lid]
            indices = np.flatnonzero(data["layout_id"] == lid)
            layout, order = verify_label_context(source_plan, entry, data, indices)
            powers = np.unique(data["total_power_dbm"][indices])
            c = physical_config(source_plan, entry, float(powers[0]), samples)
            entropy = [c["root_seed"], lid, AUDIT_STREAM, seed]
            bank = build_bank(c, layout, np.random.default_rng(np.random.SeedSequence(entropy)))
            for power in powers:
                physical = dict(c, total_dbm=float(power))
                group = indices[data["total_power_dbm"][indices] == power]
                pool_size = len(predictions) * len(group)
                for name, allocations in predictions.items():
                    for i in group:
                        xi = float(data["xi"][i])
                        y = physics_allocation(allocations[i], physical["bounds"])
                        summary, screen = assess_allocation(
                            physical, layout, bank, order, y, xi, pool_size, alpha_family,
                        )
                        _, z = original_allocation(
                            {"mb": y[0], "public": y[1:3][order].tolist(), "s": y[3]},
                            order, physical["bounds"],
                        )
                        public_lower = np.empty(2)
                        public_lower[order] = screen["public_rate_lower_sic_order"]
                        rows.append({
                            "model": name, "split": role, "layout_id": lid,
                            "total_power_dbm": float(power), "xi": xi,
                            "allocation_original_order": y.tolist(), "warm_start_pso_sic": z.tolist(),
                            "bob_rate": summary["bob_mean_rate_bit_s_hz"],
                            "dep_gamma": summary["dep_min"],
                            "dep_empirical": screen["dep_empirical_min"],
                            "dep_lower": screen["dep_uniform_lower"],
                            "dep_lower_margin": screen["dep_lower_margin"],
                            "bob_outage_upper": screen["bob_outage_upper"],
                            "public_rate_lower_original_order": public_lower.tolist(),
                            "bound_checks_pass": screen["accepted"],
                            "failed_checks": [key for key, ok in screen["checks"].items() if not ok],
                        })
            print(f"Audit {role} layout={lid}: {len(indices)} cases x {len(predictions)} allocations", flush=True)
            del bank
    return rows, {
        "samples_per_layout": samples, "bank_seed": seed, "stream": AUDIT_STREAM,
        "alpha_total": alpha, "layout_power_families": n_families, "alpha_per_family": alpha_family,
        "rows": len(rows), "elapsed_seconds": time.perf_counter() - started,
    }
