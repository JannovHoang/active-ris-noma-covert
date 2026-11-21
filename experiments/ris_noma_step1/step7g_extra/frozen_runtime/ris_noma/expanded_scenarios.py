"""Reproducible K=2 layouts and an explicit layout/power/DEP experiment plan."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import platform

import numpy as np
import scipy

from .config import validate_config
from .dataset import SPLITS, validate_partition
from .geometry import sample_layout
from .learning_data import sha256
from .random_streams import make_rng


ROOT = Path(__file__).resolve().parents[1]
FORMAT = "expanded_scenario_graph_dataset_k2_v1"
STREAMS = {"receiver_geometry": 14, "label": 15, "validation": 16, "heldout": 17, "pso": 18}
TRACKED = (
    "run_step6a.py", "ris_noma/expanded_scenarios.py", "ris_noma/expanded_data.py",
    "ris_noma/config.py", "ris_noma/geometry.py", "ris_noma/random_streams.py",
    "ris_noma/channels.py", "ris_noma/metrics.py", "ris_noma/dep.py",
    "ris_noma/evaluator.py", "ris_noma/objective.py", "ris_noma/pso.py",
    "ris_noma/validation.py", "ris_noma/cross_xi.py", "ris_noma/graph_data.py",
    "ris_noma/dataset.py", "ris_noma/learning_data.py",
)


def scenario_rng(base, layout_id, role, *details):
    entropy = [base["root_seed"], layout_id, STREAMS[role], *details]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def make_plan(base, settings, experiment):
    base = validate_config(deepcopy(base))
    experiment = deepcopy(experiment)
    required = {
        "splits", "total_power_dbm", "position_half_width_m", "pso_seeds",
        "search_dep_margin", "label_samples", "validation_samples", "heldout_samples",
        "alpha", "validation_seed", "heldout_seed",
    }
    if set(experiment) != required or base["n_public"] != 2:
        raise ValueError("Step 6a needs the declared pilot fields and K=2 physics")
    partition = {"purpose": "pilot", "splits": experiment["splits"]}
    if set(partition["splits"]) != set(SPLITS) or any(not partition["splits"][role] for role in SPLITS):
        raise ValueError("Declare at least one layout for each split")
    ids = [layout_id for role in SPLITS for layout_id in partition["splits"][role]]
    roles = validate_partition(partition, ids)
    if min(ids) < 200:
        raise ValueError("Use new pilot layout IDs in [200, 9999]")

    powers = experiment["total_power_dbm"]
    if (not powers or any(type(power) not in (int, float) or not np.isfinite(power) for power in powers)
            or len(set(powers)) != len(powers)):
        raise ValueError("Declare distinct finite total-power levels in dBm")
    for power in powers:
        validate_config(dict(base, total_dbm=power))
    widths = experiment["position_half_width_m"]
    if set(widths) != {"bob", "willie"} or any(not np.isfinite(w) or w < 0 for w in widths.values()):
        raise ValueError("Bob and Willie position half-widths must be finite and nonnegative")
    seeds = experiment["pso_seeds"]
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Declare distinct nonnegative PSO seeds")
    for key in ("label_samples", "validation_samples", "heldout_samples"):
        if type(experiment[key]) is not int or experiment[key] < 2:
            raise ValueError(f"{key} must be an integer >= 2")
    for key in ("validation_seed", "heldout_seed"):
        if type(experiment[key]) is not int or experiment[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if not 0 < experiment["alpha"] < 1:
        raise ValueError("Alpha must lie in (0, 1)")
    xis = base["requirements"]["xi"]
    margin = experiment["search_dep_margin"]
    if (len(set(xis)) != len(xis) or not np.isfinite(margin) or margin < 0
            or any(xi + margin >= 1 for xi in xis)):
        raise ValueError("DEP targets must be distinct and each target plus margin must be below one")

    layouts = []
    for layout_id in sorted(ids):
        rng = scenario_rng(base, layout_id, "receiver_geometry")
        positions = {
            node: (np.asarray(base["positions"][node]) + rng.uniform(-widths[node], widths[node], 2)).tolist()
            for node in ("bob", "willie")
        }
        physical = deepcopy(base)
        physical["positions"].update(positions)
        entry = {"layout_id": layout_id, "split": roles[layout_id], "positions": positions}
        try:
            layout = sample_layout(physical, make_rng(base["root_seed"], layout_id, "geometry"))
        except ValueError as error:
            entry.update(geometry_valid=False, public_xy_original_order=None, geometry_error=str(error))
        else:
            entry.update(geometry_valid=True, public_xy_original_order=layout.public_xy.tolist())
        layouts.append(entry)
    planned_cases = len(layouts) * len(powers) * len(xis)
    return {
        "step": "6a", "format": FORMAT, "purpose": "development_pilot",
        "base_config": base, "experiment": experiment, "partition": partition,
        "layouts": layouts, "pso_settings": asdict(settings), "streams": STREAMS,
        "planned_cases": planned_cases,
        "max_pso_evaluations": planned_cases * len(seeds) * settings.swarm_size * settings.sweeps,
        "alpha_per_power_layout_stage": experiment["alpha"] / (2 * len(layouts) * len(powers)),
        "code_sha256": {name: sha256(ROOT / name) for name in TRACKED},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "case_key": ["layout_id", "total_power_dbm", "xi"],
        "label_policy": "select on validation, freeze all selections, record independent heldout",
    }


def physical_config(plan, entry, power, samples):
    config = deepcopy(plan["base_config"])
    config.update(layout_id=entry["layout_id"], total_dbm=power, n_fading=samples)
    config["positions"].update(entry["positions"])
    return config


def reproduce_layout(config, entry):
    layout = sample_layout(config, make_rng(config["root_seed"], entry["layout_id"], "geometry"))
    if not np.allclose(layout.public_xy, entry["public_xy_original_order"], rtol=0, atol=1e-12):
        raise ValueError("Saved public positions differ from the reconstructed layout")
    return layout
