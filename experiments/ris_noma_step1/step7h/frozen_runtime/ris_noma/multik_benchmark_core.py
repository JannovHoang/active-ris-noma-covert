"""7d search, train-only SIC reference and independent bank assessments."""
from dataclasses import replace
from pathlib import Path
import time

import numpy as np

from .candidate_archive import CandidateArchive, position_key
from .channels import public_order
from .config import dbm_to_w
from .geometry_los import channel_context, pso_rng
from .learning_data import read_json, sha256
from .metrics import power_budget
from .multik_core import MultiKObjective, decode_allocation, original_allocation, physical_allocation
from .multik_data_core import at_target
from .stable_assessment import raw_assessment
from .multik_dataset import read_committed, receipt_path, write_json
from .multik_graph import statistical_graph
from .pso import PSOSettings, optimize

METHODS = ("full_pso", "random_short_pso", "lookup_sic", "lookup_sic_short_pso", "gnn", "mlp", "gnn_short_pso", "mlp_short_pso")
SELECTORS = ("search_best", "independent_bank")
_MODELS = {}


def read_group(output, phase, job):
    output = Path(output)
    record = read_committed(output / phase / (job["id"] + ".json"))
    if record is not None and record.get("plan_sha256") != sha256(output / "plan.json"):
        origin = read_json(output / "plan.json").get("search_origin")
        name = phase + "/" + job["id"] + ".json"
        if (phase != "search" or origin is None or record.get("plan_sha256") != origin["original_plan_sha256"]
                or sha256(output / name) != origin["search_files_sha256"].get(name)):
            raise ValueError("Cached 7d group belongs to another frozen plan")
    return record


def table_policy(train_rows, source, margin):
    """Only train labels determine the P/DEP lookup; all K share one table."""
    groups = {}
    for row in train_rows:
        groups.setdefault((row["power_dbm"], row["xi"]), []).append([row["allocation"][0], row["allocation"][-1]])
    return {"fit_role": "train", "sic_rate_margin": margin,
        "parameters": [{"power_dbm": power, "xi": xi, "mb": float(np.mean(values, axis=0)[0]), "s": float(np.mean(values, axis=0)[1])}
                       for (power, xi), values in sorted(groups.items())],
        "policy": "train P/DEP means; minimum SIC ceiling powers at QoS+margin; residual to weakest stream; clip mb by K-dependent analytical ceiling cap",
        "finite_snr_feasibility_guaranteed": False}


def lookup_allocation(policy, c, order, power, xi):
    entry = next((v for v in policy["parameters"] if (v["power_dbm"], v["xi"]) == (power, xi)), None)
    if entry is None:
        raise ValueError("Lookup has no TRAIN entry at this power/DEP; no interpolation is hidden")
    gamma = np.expm1(np.log(2.) * (np.asarray(c["requirements"]["qos_public"])[order] + policy["sic_rate_margin"]))
    k2 = c["kappa"] ** 2
    cap = (1 + k2) / float(np.prod(1 + gamma)) - k2
    if cap < c["bounds"]["mb"][0]:
        return None
    mb = min(entry["mb"], cap, c["bounds"]["mb"][1])
    public = np.zeros(c["n_public"])
    interference = mb + k2
    for j in range(len(public) - 1, -1, -1):
        public[j] = gamma[j] * interference
        interference += public[j]
    remaining = 1 - mb - public.sum()
    if remaining < -1e-10:
        raise ArithmeticError("SIC allocation cap and recursion disagree")
    public[0] += max(0., remaining)
    return original_allocation(mb, public, entry["s"], order, c["bounds"])[0]


def load_net(root, ref):
    from .learning_train import configure_runtime
    from .multik_learning_models import load_model
    key = (str(Path(root).resolve()), ref["sha256"])
    if key not in _MODELS:
        configure_runtime(0, 1, "cpu")
        path = Path(root) / ref["path"]
        if sha256(path) != ref["sha256"]:
            raise ValueError("Frozen model changed")
        _MODELS[key] = load_model(path)[0]
    return _MODELS[key]


def search_pool(objective, c, order, target, settings, random, initial=None):
    protected = None if initial is None else initial[0]
    archive = CandidateArchive(target, [.005, .01], 5, protected)
    def evaluate(z):
        outcome = objective.evaluate(z, target)
        archive.observe(z, outcome)
        return outcome
    started = time.perf_counter()
    result = optimize(evaluate, c["n_public"] + 1, settings, random, initial)
    if archive.count != result.evaluations or position_key(result.position) != archive.top[0]["key"]:
        raise ArithmeticError("PSO/archive winner mismatch")
    candidates = []
    for item in archive.pool():
        mb, public, s = decode_allocation(item["position"], c["n_public"], c["bounds"])
        y, _ = original_allocation(mb, public, s, order, c["bounds"])
        candidates.append({"candidate_id": f"e{item['evaluation']:06d}", "allocation": y.tolist(),
            "evaluation": item["evaluation"], "search_rate": item["outcome"].score,
            "search_dep_gamma": item["outcome"].metrics["dep_min"], "search_feasible": item["outcome"].feasible})
    return {"candidates": candidates, "raw_candidate_id": f"e{archive.top[0]['evaluation']:06d}",
        "pso_seconds": time.perf_counter() - started, "objective_evaluations": result.evaluations,
        "feasible_evaluations": result.feasible_evaluations, "history": result.history}


def diagnostic(c, layout, bank, order, allocation):
    y = physical_allocation(allocation, c["n_public"], c["bounds"])
    budget = power_budget(c, layout.loss_ar, y[-1])
    U = budget["pa_w"] * bank.signal_gain_tilde[:, 0] / (
        float(dbm_to_w(c["noise_dbm"]["ris"])) * bank.noise_gain_tilde[:, 0]
        + float(dbm_to_w(c["noise_dbm"]["bob"])) / budget["rho2"])
    ceiling = float(np.log2(1 + y[0] / c["kappa"] ** 2))
    rate = float(np.log2(1 + y[0] * U / (c["kappa"] ** 2 * U + 1)).mean())
    return {"median_kappa2_U_bob": float(np.median(c["kappa"] ** 2 * U)),
        "median_additive_noise_fraction_bob": float(np.median(1 / (1 + c["kappa"] ** 2 * U))),
        "bob_rate_to_own_hardware_ceiling": rate / ceiling, "bob_ceiling": ceiling, "bob_rate": rate}


def search_job(arguments):
    output, job, training_root = arguments
    output = Path(output)
    path = output / "search" / (job["id"] + ".json")
    if read_group(output, "search", job) is not None:
        return job["id"]
    plan = read_json(output / "plan.json")
    source = read_json(output / "source_physics_plan.json")
    dataset_plan = read_json(output / "source_dataset_plan.json")
    config = plan["config"]
    try:
        c, layout, bank, metadata = channel_context(plan, source, dataset_plan, job, "search")
    except ValueError as error:
        if not str(error).startswith("Layout bị loại:"):
            raise
        record = {"job": job, "geometry_valid": False, "reason": str(error), "pools": [], "objective_evaluations": 0,
            "plan_sha256": sha256(output / "plan.json")}
        write_json(path, record); write_json(receipt_path(path), {"sha256": sha256(path)})
        return job["id"]
    order = public_order(bank)
    eligible = MultiKObjective(c, layout, bank, order).layout_eligible
    pools = []; evaluations = 0; fixed_probes = []
    if job["stage"] == "calibration":
        methods, margins = ("full_pso",), [config["calibration_margin"]]
        replicates = [(None, config["pso_seeds"][0])]
        xis = config["calibration_xi"]
    else:
        methods, margins, xis = METHODS, config["search_margins"], config["xi"]
        replicates = list(zip(config["training_seeds"], config["pso_seeds"]))
    models = {}
    if job["stage"] == "benchmark":
        from .multik_models import predict
        for seed, _ in replicates:
            for kind in ("gnn", "mlp"):
                models[(kind, seed)] = load_net(training_root, plan["models"][f"{kind}_s{seed}"])
        warm_graph = statistical_graph(c, layout, bank, order, xis[0])
        for model in models.values(): predict(model, [warm_graph], plan["normalization"])
    for pi, power in enumerate(config["power_dbm"]):
        physical = dict(c, total_dbm=power)
        probe = [.28, *([.72 / c["n_public"]] * c["n_public"]), .5]
        fixed_probes.append({"power_dbm": power, "allocation": probe,
            "purpose": "fixed physical diagnostic only; not assumed feasible", **diagnostic(physical, layout, bank, order, probe)})
        started = time.perf_counter(); objective = MultiKObjective(physical, layout, bank, order)
        objective_seconds = time.perf_counter() - started
        for ti, xi in enumerate(xis):
            started = time.perf_counter(); graph = statistical_graph(physical, layout, bank, order, xi)
            graph_seconds = time.perf_counter() - started
            for training_seed, seed in replicates:
                forecasts = {}; forecast_seconds = {}
                if job["stage"] == "benchmark":
                    for kind in ("gnn", "mlp"):
                        started = time.perf_counter()
                        forecasts[kind] = predict(models[(kind, training_seed)], [graph], plan["normalization"])[0]
                        forecast_seconds[kind] = time.perf_counter() - started + graph_seconds
                    started = time.perf_counter()
                    forecasts["lookup_sic"] = lookup_allocation(plan["lookup_policy"], physical, order, power, xi)
                    forecast_seconds["lookup_sic"] = time.perf_counter() - started
                rotation = (job["layout_id"] + pi + ti + seed) % len(methods)
                rotated = methods[rotation:] + methods[:rotation]
                for margin in margins:
                    for method in rotated:
                        source_kind = method.removesuffix("_short_pso") if method.endswith("_short_pso") else method
                        is_pso = method.endswith("pso")
                        prefix = forecast_seconds.get(source_kind, 0.)
                        common = {"power_dbm": power, "xi": xi, "search_margin": margin, "method": method,
                            "training_seed": training_seed, "pso_seed": seed,
                            "objective_cache_seconds": objective_seconds if is_pso else 0.,
                            "input_and_forward_seconds": prefix, "search_bank_seconds": metadata["bank_generation_seconds"]}
                        if source_kind in ("gnn", "mlp"):
                            common["model_checkpoint_sha256"] = plan["models"][f"{source_kind}_s{training_seed}"]["sha256"]
                        initial = None
                        if is_pso:
                            if method not in ("full_pso", "random_short_pso"):
                                y = forecasts[source_kind]
                                if y is not None:
                                    _, center = original_allocation(y[0], np.asarray(y[1:-1])[order], y[-1], order, c["bounds"])
                                    initial = center[None, :]
                            common["warm_start_injected"] = initial is not None
                            common["warm_start_allocation"] = None if initial is None else np.asarray(forecasts[source_kind]).tolist()
                            budget = config["full_evaluations"] if method == "full_pso" else config["short_evaluations"]
                            settings = PSOSettings(**dict(plan["pso_settings"], swarm_size=30, sweeps=budget // 30))
                            result = search_pool(objective, physical, order, xi + margin, settings,
                                pso_rng(c["root_seed"], job, seed, pi, ti), initial)
                            evaluations += result["objective_evaluations"]
                        else:
                            y = forecasts[method]
                            result = {"candidates": [] if y is None else [{"candidate_id": "direct", "allocation": np.asarray(y).tolist(), "evaluation": 0}],
                                "raw_candidate_id": None if y is None else "direct", "pso_seconds": 0., "objective_evaluations": 0,
                                "feasible_evaluations": 0, "history": []}
                        pool = {**common, **result}
                        pool["search_return_seconds_cached_bank"] = prefix + pool["objective_cache_seconds"] + pool["pso_seconds"]
                        if result["candidates"]:
                            winner = next(p for p in result["candidates"] if p["candidate_id"] == result["raw_candidate_id"])
                            pool["search_diagnostic"] = diagnostic(physical, layout, bank, order, winner["allocation"])
                        pools.append(pool)
    record = {"job": job, "geometry_valid": True, "layout_eligible_on_search_bank": eligible,
        "metadata": metadata, "order": order.tolist(), "pools": pools, "fixed_probes": fixed_probes, "objective_evaluations": evaluations,
        "plan_sha256": sha256(output / "plan.json")}
    write_json(path, record); write_json(receipt_path(path), {"sha256": sha256(path)})
    return job["id"]


def select_job(arguments):
    output, job = arguments
    output = Path(output); path = output / "selection" / (job["id"] + ".json")
    if read_group(output, "selection", job) is not None: return job["id"]
    if not (output / "search_frozen.json").is_file():
        raise ValueError("Freeze every search pool before reading selection banks")
    plan = read_json(output / "plan.json")
    search = read_group(output, "search", job)
    rows = []; cache = {}; unique_seconds = {}; reviewed_pools = []
    if search["geometry_valid"]:
        c, layout, bank, metadata = channel_context(plan, read_json(output / "source_physics_plan.json"),
            read_json(output / "source_dataset_plan.json"), job, "selection")
        family = job["selection_family"]
        alpha = plan["config"]["alpha"] / (2 * len(plan["jobs"]))
        for index, pool in enumerate(search["pools"]):
            scored = []; seconds = 0.
            for candidate in pool["candidates"]:
                key = (pool["power_dbm"], tuple(candidate["allocation"]))
                if key not in cache:
                    started = time.perf_counter()
                    raw = raw_assessment(dict(c, total_dbm=pool["power_dbm"]), layout, bank,
                        np.asarray(search["order"], int), candidate["allocation"], family, alpha)
                    cache[key] = raw; unique_seconds[key] = time.perf_counter() - started
                raw = cache[key]; seconds += unique_seconds[key]
                scored.append({**candidate, "physical": raw, **at_target(raw, pool["xi"])})
            reviewed_pools.append({"pool_index": index, "candidates": scored})
            raw_choice = next((p for p in scored if p["candidate_id"] == pool["raw_candidate_id"]), None)
            eligible = [p for p in scored if p["passed"]]
            selected = min(eligible, key=lambda p: (-p["physical"]["bob_rate"], p["candidate_id"])) if eligible else None
            for selector, candidate in (("search_best", raw_choice), ("independent_bank", selected)):
                cached_seconds = pool["search_return_seconds_cached_bank"] + (seconds if selector == "independent_bank" else 0.)
                rows.append({"pool_index": index, **{key: pool[key] for key in ("power_dbm", "xi", "search_margin", "method", "training_seed", "pso_seed", "objective_evaluations")},
                    "selector": selector, "returned": candidate is not None,
                    "candidate_id": None if candidate is None else candidate["candidate_id"],
                    "allocation": None if candidate is None else candidate["allocation"],
                    "selection_passed": None if candidate is None else candidate["passed"],
                    "selection_bob_rate": None if candidate is None else candidate["physical"]["bob_rate"],
                    "selection_candidate_count": len(scored), "selection_candidate_seconds": seconds if selector == "independent_bank" else 0.,
                    "return_seconds_cached_banks": cached_seconds,
                    "return_seconds_with_bank_generation": cached_seconds + pool["search_bank_seconds"] + (metadata["bank_generation_seconds"] if selector == "independent_bank" else 0.)})
    else: metadata = None
    result = {"job": job, "rows": rows, "reviewed_pools": reviewed_pools, "candidate_assessments": len(cache),
        "metadata": metadata, "search_frozen_sha256": sha256(output / "search_frozen.json"),
        "plan_sha256": sha256(output / "plan.json")}
    write_json(path, result); write_json(receipt_path(path), {"sha256": sha256(path)})
    return job["id"]


def audit_job(arguments):
    output, job = arguments
    output = Path(output); path = output / "audit" / (job["id"] + ".json")
    if read_group(output, "audit", job) is not None: return job["id"]
    if not (output / "selection_frozen.json").is_file():
        raise ValueError("Freeze every selection before reading final audit banks")
    plan = read_json(output / "plan.json")
    search = read_group(output, "search", job)
    selected = read_group(output, "selection", job)
    cache = {}; rows = []
    if search["geometry_valid"]:
        c, layout, bank, metadata = channel_context(plan, read_json(output / "source_physics_plan.json"),
            read_json(output / "source_dataset_plan.json"), job, "audit")
        family = job["audit_family"]; alpha = plan["config"]["alpha"] / (2 * len(plan["jobs"]))
        for row in selected["rows"]:
            if row["returned"]:
                key = (row["power_dbm"], tuple(row["allocation"]))
                if key not in cache:
                    cache[key] = raw_assessment(dict(c, total_dbm=row["power_dbm"]), layout, bank,
                        np.asarray(search["order"], int), row["allocation"], family, alpha)
                raw = cache[key]; outcome = {**raw, **at_target(raw, row["xi"])}
            else:
                outcome = {"bob_rate": 0., "gated_bob_rate": 0., "passed": False, "failed_checks": ["no_candidate_returned"]}
            rows.append({**job, **row, **outcome})
    else: metadata = None
    result = {"job": job, "rows": rows, "unique_assessments": len(cache), "metadata": metadata,
        "selection_frozen_sha256": sha256(output / "selection_frozen.json"),
        "plan_sha256": sha256(output / "plan.json")}
    write_json(path, result); write_json(receipt_path(path), {"sha256": sha256(path)})
    return job["id"]
