"""Step 6d. Freeze paired PSO prefixes before scoring held-out layouts."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from multiprocessing import get_context
from pathlib import Path
import platform
import time

from filelock import FileLock
import numpy as np
import scipy
import torch

from .benchmark_pso import commit_record, load_record, predict_start, settings_for, validate_config_5c, warm_positions
from .budget_analysis import METHODS, report
from .channels import build_bank, public_order
from .expanded_benchmark import CODE_FILES as PRIOR_CODE, check_benchmark, select_models, source_hashes
from .expanded_data import save_json
from .expanded_learning import validate_source
from .expanded_scenarios import ROOT, physical_config, reproduce_layout, scenario_rng
from .geometry import sample_layout
from .graph_data import original_allocation, statistical_graph
from .learning_data import read_json, sha256
from .learning_physics import assess_allocation
from .learning_train import configure_runtime, load_checkpoint
from .objective import CachedObjective, decode_k2
from .pso import better, optimize
from .random_streams import make_rng


STREAMS = {"search": 23, "optimizer": 24, "warm_jitter": 25, "audit": 26, "bootstrap": 27}
CODE_FILES = (*PRIOR_CODE, "run_step6d.py", "ris_noma/budget_benchmark.py", "ris_noma/budget_analysis.py")


def new_layout(source, layout_id):
    base = source["base_config"]
    experiment = source["experiment"]
    rng = scenario_rng(base, layout_id, "receiver_geometry")
    positions = {}
    for name, width in experiment["position_half_width_m"].items():
        positions[name] = (np.asarray(base["positions"][name]) + rng.uniform(-width, width, 2)).tolist()
    entry = {"layout_id": layout_id, "positions": positions}
    c = physical_config(source, entry, experiment["total_power_dbm"][0], experiment["label_samples"])
    try:
        layout = sample_layout(c, make_rng(c["root_seed"], layout_id, "geometry"))
    except ValueError as error:
        return {**entry, "geometry_valid": False, "geometry_error": str(error)}
    return {**entry, "geometry_valid": True, "public_xy_original_order": layout.public_xy.tolist()}


def benchmark_hashes(benchmark):
    return {name: sha256(benchmark / name) for name in ("benchmark_plan.json", "completion.json")}


def inspect_budget(dataset, training, benchmark, config, workers):
    required = {"layout_ids", "budgets", "primary_budget", "cross_budget_reference", "bootstrap_repetitions"}
    if set(config) != required or type(workers) is not int or workers < 1:
        raise ValueError("Invalid step 6d configuration or worker count")
    check_benchmark(benchmark)
    prior = read_json(benchmark / "benchmark_plan.json")
    c = validate_config_5c(prior["config"])
    hashes = source_hashes(dataset, training)
    if prior["sources_sha256"] != hashes:
        raise ValueError("Step 6c must use these exact 6a and 6b inputs")
    configure_runtime(0, c["threads"], "cpu")
    _, schema = validate_source(dataset)
    source = read_json(dataset / "plan.json")
    selected, _ = select_models(dataset, training, hashes, source["base_config"]["bounds"], schema)
    if prior["selected_checkpoints"] != selected:
        raise ValueError("The recorded 6c checkpoint choices differ from 6b")
    seen = {row["layout_id"] for row in source["layouts"]} | {row["layout_id"] for row in prior["cases"]}
    ids = config["layout_ids"]
    if (not isinstance(ids, list) or len(ids) < 2 or any(type(i) is not int or i < 200 for i in ids)
            or len(set(ids)) != len(ids) or set(ids) & seen):
        raise ValueError("Declare at least two distinct layout IDs outside all 6a/6c splits")
    budgets = config["budgets"]
    primary, cross = config["primary_budget"], config["cross_budget_reference"]
    repetitions = config["bootstrap_repetitions"]
    if (not isinstance(budgets, list) or not budgets
            or any(type(n) is not int or n <= 0 or n % c["swarm_size"] for n in budgets)
            or budgets != sorted(set(budgets)) or budgets[-1] != c["swarm_size"] * c["short_sweeps"]
            or type(primary) is not int or type(cross) is not int
            or primary not in budgets or cross not in budgets or cross <= primary
            or type(repetitions) is not int or repetitions < 1000):
        raise ValueError("Declare increasing whole-sweep budgets ending at the 6c short budget")
    powers = source["experiment"]["total_power_dbm"]
    xis = source["base_config"]["requirements"]["xi"]
    pairs_per_layout = len(powers) * len(xis) * len(c["pso_seeds"])
    pairs = len(ids) * pairs_per_layout
    plan = {
        "step": "6d", **config, "layout_ids": sorted(ids),
        "layouts": [new_layout(source, lid) for lid in sorted(ids)],
        "scope": "new K=2 layouts from the 6a geometry distribution, conditional on layout eligibility",
        "sources_sha256": hashes, "source_6c_sha256": benchmark_hashes(benchmark),
        "selected_checkpoint": selected["gnn"], "methods": list(METHODS), "search_config": c,
        "power_dbm": powers, "xi": xis, "search_samples_per_layout": source["experiment"]["label_samples"],
        "audit_samples_per_layout": c["audit_samples"], "audit_alpha": c["audit_alpha"],
        "audit_alpha_per_family": c["audit_alpha"] / (len(ids) * len(powers)),
        "audit_family_size": len(xis) * len(c["pso_seeds"]) * len(METHODS) * len(budgets),
        "workers": workers, "streams": STREAMS,
        "paired_trajectories_planned": pairs, "paired_trajectories_per_layout": pairs_per_layout,
        "psovals_planned": pairs * len(METHODS) * budgets[-1],
        "audit_rows_planned": pairs * len(METHODS) * len(budgets),
        "timing_scope": "search bank and objective cache prepared; features, inference, warm particles and PSO included",
        "code_sha256": {name: sha256(ROOT / name) for name in CODE_FILES},
        "versions": {"system": platform.system(), "machine": platform.machine(),
                     "python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "torch": str(torch.__version__)},
    }
    return plan, source


def stage_rng(root_seed, lid, stream, *indices):
    return np.random.default_rng(np.random.SeedSequence([root_seed, lid, STREAMS[stream], *indices]))


def capture_budgets(objective, settings, rng, initial, budgets, wall_started, cpu_started):
    """Observe the existing optimizer at sweep boundaries without another evaluation."""
    observed, count, best, position = [], 0, None, None
    marks = set(budgets)

    def record(z):
        nonlocal count, best, position
        outcome = objective(z)
        if better(outcome, best):
            best, position = outcome, z.copy()
        count += 1
        if count in marks:
            observed.append({"budget": count, "z": position.tolist(),
                             "search_feasible": bool(best.feasible), "search_bob_rate": float(best.score),
                             "search_wall_seconds": time.perf_counter() - wall_started,
                             "search_cpu_seconds": time.process_time() - cpu_started})
        return outcome

    result = optimize(record, 3, settings, rng, initial)
    if (count != result.evaluations or [row["budget"] for row in observed] != budgets
            or not np.array_equal(observed[-1]["z"], result.position)):
        raise ArithmeticError("Budget capture differs from the optimizer")
    for row in observed:
        history = result.history[row["budget"] // settings.swarm_size - 1]
        if row["search_feasible"] != history["gbest_feasible"] or row["search_bob_rate"] != history["gbest_score"]:
            raise ArithmeticError("Captured candidate differs from the optimizer sweep history")
    return observed


def search_layout(entry, plan, source, checkpoint_path):
    lid = entry["layout_id"]
    if not entry["geometry_valid"]:
        return {"layout_id": lid, "status": "invalid_geometry", "trajectories": []}
    c = plan["search_config"]
    physical = physical_config(source, entry, plan["power_dbm"][0], plan["search_samples_per_layout"])
    root = physical["root_seed"]
    layout = reproduce_layout(physical, entry)
    bank = build_bank(physical, layout, stage_rng(root, lid, "search"))
    order = public_order(bank)
    means = bank.signal_gain_tilde.mean(axis=0)
    if not means[0] > np.max(means[1:-1]):
        return {"layout_id": lid, "status": "bob_not_strongest", "trajectories": []}
    if sha256(checkpoint_path) != plan["selected_checkpoint"]["sha256"]:
        raise ValueError("The frozen GNN checkpoint changed")
    configure_runtime(0, c["threads"], "cpu")
    model = load_checkpoint(checkpoint_path)
    settings = settings_for(c, "pso_short")
    trajectories = []
    for power_idx, power in enumerate(plan["power_dbm"]):
        params = dict(physical, total_dbm=power)
        objective = CachedObjective(params, layout, bank, order)
        for xi_idx, xi in enumerate(plan["xi"]):
            target = xi + c["search_dep_margin"]
            for seed_idx, seed in enumerate(c["pso_seeds"]):
                methods = METHODS if (lid + power_idx + xi_idx + seed_idx) % 2 == 0 else METHODS[::-1]
                for method in methods:
                    wall_started, cpu_started = time.perf_counter(), time.process_time()
                    initial = None
                    if method == "gnn_pso_short":
                        graph = statistical_graph(params, layout, bank, order, xi)
                        _, center = predict_start(*model, graph, order, params["bounds"])
                        initial = warm_positions(center, c["warm_particles"], c["warm_sigma"],
                                                 stage_rng(root, lid, "warm_jitter", seed, power_idx, xi_idx))
                    rng = stage_rng(root, lid, "optimizer", seed, power_idx, xi_idx)
                    snapshots = capture_budgets(lambda z: objective.evaluate(z, target), settings, rng, initial,
                                                plan["budgets"], wall_started, cpu_started)
                    for row in snapshots:
                        mb, public, s = decode_k2(row["z"], params["bounds"])
                        allocation, _ = original_allocation({"mb": mb, "public": public, "s": s}, order, params["bounds"])
                        row["allocation"] = allocation.tolist()
                    trajectories.append({"total_power_dbm": power, "xi": xi, "pso_seed": seed,
                                         "method": method, "snapshots": snapshots})
    return {"layout_id": lid, "status": "eligible", "sic_order_original_ids": order.tolist(), "trajectories": trajectories}


def audit_layout(entry, plan, source, search, freeze_hash):
    c = plan["search_config"]
    lid = entry["layout_id"]
    base = physical_config(source, entry, plan["power_dbm"][0], plan["audit_samples_per_layout"])
    layout = reproduce_layout(base, entry)
    bank = build_bank(base, layout, stage_rng(base["root_seed"], lid, "audit", c["audit_seed"]))
    order = np.asarray(search["sic_order_original_ids"], int)
    qos = np.asarray(base["requirements"]["qos_public"])[order]
    rows, cache = [], {}
    for trajectory in search["trajectories"]:
        power, xi = trajectory["total_power_dbm"], trajectory["xi"]
        physical = dict(base, total_dbm=power)
        identity = {key: trajectory[key] for key in ("total_power_dbm", "xi", "pso_seed", "method")}
        for point in trajectory["snapshots"]:
            key = power, xi, tuple(point["allocation"])
            if key not in cache:
                info, screen = assess_allocation(physical, layout, bank, order, point["allocation"], xi,
                                                 plan["audit_family_size"], plan["audit_alpha_per_family"])
                rate, passed = info["bob_mean_rate_bit_s_hz"], screen["accepted"]
                cache[key] = {
                    "audit_pass": passed, "bob_rate": rate, "gated_bob_rate": rate if passed else 0.,
                    "dep_gamma": info["dep_min"], "dep_empirical": screen["dep_empirical_min"],
                    "dep_lower": screen["dep_uniform_lower"],
                    "qos_lower_margin": float(np.min(np.asarray(screen["public_rate_lower_sic_order"]) - qos)),
                    "bob_outage_upper": screen["bob_outage_upper"],
                    "failed_checks": [name for name, ok in screen["checks"].items() if not ok],
                }
            rows.append({"layout_id": lid, **identity, **point, **cache[key]})
    return {"layout_id": lid, "search_frozen_sha256": freeze_hash, "rows": rows, "unique_audit_evaluations": len(cache)}


def record_path(output, phase, lid):
    return output / phase / f"layout_{lid:06d}.json"


def run_phase(output, phase, worker, jobs, workers, frozen_search=False):
    saved, pending = {}, []
    for args in jobs:
        lid = args[0]["layout_id"]
        record = load_record(record_path(output, phase, lid))
        if record is None:
            if frozen_search:
                raise ValueError(f"Missing frozen search for layout {lid}")
            pending.append(args)
        elif record["layout_id"] == lid:
            saved[lid] = record
        else:
            raise ValueError(f"Incorrect layout in committed {phase} result")
    if not pending:
        return saved
    with ExitStack() as stack:
        if workers == 1:
            results = (worker(*args) for args in pending)
        else:
            pool = stack.enter_context(ProcessPoolExecutor(max_workers=min(workers, len(pending)),
                                                            mp_context=get_context("spawn")))
            futures = [pool.submit(worker, *args) for args in pending]
            results = (future.result() for future in as_completed(futures))
        for record in results:
            lid = record["layout_id"]
            commit_record(record_path(output, phase, lid), record)
            saved[lid] = record
            print(f"{phase} layout {lid}: {len(saved)}/{len(jobs)} committed", flush=True)
    return saved


def check_budget(output):
    output = output.resolve()
    hashes = read_json(output / "completion.json")["sha256"]
    required = {"plan.json", "search_frozen.json", "summary.json", "comparison_runs.csv", "curves.csv",
                "paired_comparison.csv", "paired_by_layout.csv"}
    if not required.issubset(hashes):
        raise ValueError("Incomplete step 6d output")
    for name, digest in hashes.items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Step 6d result changed: {name}")
    result, plan, frozen = (read_json(output / name) for name in ("summary.json", "plan.json", "search_frozen.json"))
    planned = set(plan["layout_ids"])
    eligible = set(result["eligible_layout_ids"])
    if (result["step"] != "6d" or not result["complete"] or not eligible.issubset(planned)
            or frozen["plan_sha256"] != hashes["plan.json"]
            or result["planned_layouts"] != len(planned) or result["eligible_layouts"] != len(eligible)
            or set(frozen["sha256"]) != {record_path(Path(), "search", lid).as_posix() for lid in planned}
            or any(hashes.get(name) != digest for name, digest in frozen["sha256"].items())
            or any(record_path(Path(), "audit", lid).as_posix() not in hashes for lid in eligible)
            or result["paired_trajectories"] != len(eligible) * plan["paired_trajectories_per_layout"]
            or result["psovals"] != result["paired_trajectories"] * len(METHODS) * plan["budgets"][-1]
            or result["audit_rows"] != result["paired_trajectories"] * len(METHODS) * len(plan["budgets"])):
        raise ValueError("Step 6d output does not match its frozen plan")
    return result


def run_budget(dataset, training, benchmark, output, config, workers=4, resume=False):
    dataset, training, benchmark, output = (path.resolve() for path in (dataset, training, benchmark, output))
    for source in (dataset, training, benchmark):
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("Input and output directories must be separate")
    plan, source = inspect_budget(dataset, training, benchmark, config, workers)
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output.with_name(f".{output.name}.lock")), timeout=0):
        return run_locked(dataset, training, benchmark, output, plan, source, resume)


def run_locked(dataset, training, benchmark, output, plan, source, resume):
    plan_path = output / "plan.json"
    if output.exists() and any(output.iterdir()):
        if not resume or not plan_path.is_file() or read_json(plan_path) != plan:
            raise ValueError("Use a new output or --resume with the identical plan and worker count")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "completion.json").is_file():
        return check_budget(output)
    if not plan_path.is_file():
        save_json(plan_path, plan)
    checkpoint = training / plan["selected_checkpoint"]["path"]
    frozen_path = output / "search_frozen.json"
    searches = run_phase(output, "search", search_layout,
                         [(entry, plan, source, checkpoint) for entry in plan["layouts"]],
                         plan["workers"], frozen_path.is_file())
    frozen = {"plan_sha256": sha256(plan_path), "sha256": {
        record_path(Path(), "search", lid).as_posix(): sha256(record_path(output, "search", lid)) for lid in sorted(searches)
    }}
    if frozen_path.is_file():
        if read_json(frozen_path) != frozen:
            raise ValueError("Frozen search changed")
    else:
        save_json(frozen_path, frozen)
    freeze_hash = sha256(frozen_path)
    eligible = [entry for entry in plan["layouts"] if searches[entry["layout_id"]]["status"] == "eligible"]
    if not eligible:
        raise ValueError("No new layouts meet the K=2 model eligibility rule")
    print(f"Search frozen for {len(eligible)} eligible layouts. Starting fresh audit.", flush=True)
    audits = run_phase(output, "audit", audit_layout,
                       [(entry, plan, source, searches[entry["layout_id"]], freeze_hash) for entry in eligible],
                       plan["workers"])
    if (any(record["search_frozen_sha256"] != freeze_hash for record in audits.values())
            or sha256(frozen_path) != freeze_hash
            or any(sha256(output / name) != digest for name, digest in frozen["sha256"].items())
            or source_hashes(dataset, training) != plan["sources_sha256"]
            or benchmark_hashes(benchmark) != plan["source_6c_sha256"]
            or sha256(checkpoint) != plan["selected_checkpoint"]["sha256"]):
        raise ValueError("The source inputs or frozen search changed during step 6d")
    rows = [row for lid in sorted(audits) for row in audits[lid]["rows"]]
    if {row["layout_id"] for row in rows} != {entry["layout_id"] for entry in eligible}:
        raise ValueError("Every eligible layout needs a completed independent audit")
    result = report(output, plan, rows)
    result["excluded_layouts"] = {lid: r["status"] for lid, r in searches.items() if r["status"] != "eligible"}
    result["unique_audit_evaluations"] = sum(r["unique_audit_evaluations"] for r in audits.values())
    save_json(output / "summary.json", result)
    save_json(output / "completion.json", {"sha256": {
        path.relative_to(output).as_posix(): sha256(path) for path in sorted(output.rglob("*"))
        if path.is_file() and path.suffix != ".tmp" and path != output / "completion.json"
        and (output / "figures") not in path.parents
    }})
    return check_budget(output)
