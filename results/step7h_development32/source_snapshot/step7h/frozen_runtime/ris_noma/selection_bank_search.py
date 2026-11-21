"""Generate all 6g pools before any independent selection sample is read."""
from dataclasses import replace
import time

import numpy as np

from .benchmark_pso import predict_start, settings_for, warm_positions
from .candidate_archive import capture_candidates
from .channels import public_order
from .dep_label_experiment import channel_context, rng
from .graph_data import statistical_graph
from .learning_train import configure_runtime, load_checkpoint
from .objective import CachedObjective


def search_layout(entry, plan, source, role, margins, checkpoint_path, upstream):
    record = {"layout_id": entry["layout_id"], "status": "invalid_geometry", "pools": [],
              "pso_evaluations": 0, "upstream_frozen_sha256": upstream}
    if not entry["geometry_valid"]:
        return record
    started = time.perf_counter()
    base, layout, bank = channel_context(plan, source, entry, f"{role}_search", plan["search_samples"])
    bank_seconds = time.perf_counter() - started
    order = public_order(bank)
    if not CachedObjective(base, layout, bank, order).layout_eligible:
        return dict(record, status="bob_not_strongest")
    record.update(status="eligible", order=order.tolist())
    c = plan["search_config"]
    configure_runtime(0, c["threads"], "cpu")
    model, checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint["feature_schema"] != plan["feature_schema"]:
        raise ValueError("Frozen GNN graph schema differs")
    warmup = statistical_graph(base, layout, bank, order, plan["xi"][0])
    predict_start(model, checkpoint, warmup, order, base["bounds"])
    branches = ("cold", "warm", "full") if role == "test" else ("cold", "warm")
    for power_index, power in enumerate(plan["power_dbm"]):
        physical = dict(base, total_dbm=power)
        started = time.perf_counter()
        objective = CachedObjective(physical, layout, bank, order)
        cache_seconds = time.perf_counter() - started
        for xi_index, xi in enumerate(plan["xi"]):
            started = time.perf_counter()
            graph = statistical_graph(physical, layout, bank, order, xi)
            allocation, center = predict_start(model, checkpoint, graph, order, physical["bounds"])
            gnn_seconds = time.perf_counter() - started
            for margin in margins:
                common = {"layout_id": entry["layout_id"], "total_power_dbm": power, "xi": xi, "search_margin": margin,
                          "search_bank_seconds": bank_seconds}
                record["pools"].append({**common, "branch": "gnn", "budget": 0, "pso_seed": None,
                    "objective_cache_seconds": 0., "search_wall_seconds": gnn_seconds,
                    "raw_candidate_id": "gnn", "candidates": [{"candidate_id": "gnn", "allocation": allocation.tolist(), "evaluation": 0}]})
                for seed_index, seed in enumerate(c["pso_seeds"]):
                    rotation = (entry["layout_id"] + power_index + xi_index + seed_index) % len(branches)
                    for branch in branches[rotation:] + branches[:rotation]:
                        started = time.perf_counter()
                        budgets = [plan["full_evaluations"]] if branch == "full" else plan["budgets"]
                        settings = replace(settings_for(c, "pso_full" if branch == "full" else "pso_short"),
                                           sweeps=budgets[-1] // c["swarm_size"])
                        initial = None
                        if branch == "warm":
                            initial = warm_positions(center, c["warm_particles"], c["warm_sigma"],
                                rng(plan, base, entry, f"{role}_jitter", seed, power_index, xi_index))
                        prefix = time.perf_counter() - started + (gnn_seconds if branch == "warm" else 0.)
                        target = xi + margin
                        points = capture_candidates(lambda z: objective.evaluate(z, target), settings,
                            rng(plan, base, entry, f"{role}_pso", seed, power_index, xi_index), initial,
                            budgets, target, plan["archive_slacks"], order, physical["bounds"], prefix)
                        record["pso_evaluations"] += budgets[-1]
                        for point in points:
                            record["pools"].append({**common, "branch": branch, "pso_seed": seed,
                                "objective_cache_seconds": cache_seconds, **point})
    return record
