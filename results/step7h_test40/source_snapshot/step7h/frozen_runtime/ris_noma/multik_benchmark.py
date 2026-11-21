"""7d development calibration and paired eight-method benchmark.

Search pools freeze globally before selection banks; all choices then freeze
before final audit banks. Existing 7b/7c data and weights are read-only.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
import itertools
import json
import multiprocessing as mp
import platform
import re

from filelock import FileLock
import numpy as np
import scipy

from .geometry_los import PROFILES, STREAMS, array_elements
from .learning_data import read_json, sha256
from .multik_data_core import at_target, geometry
from .multik_dataset import immutable, immutable_bytes, read_committed, read_jsonl, receipt_path
from .multik_benchmark_core import METHODS, SELECTORS, table_policy, search_job, select_job, audit_job, read_group

ROOT = Path(__file__).resolve().parents[1]
NUMERIC_FILES = ("run_step7d.py", "ris_noma/geometry_los.py", "ris_noma/multik_benchmark_core.py",
                 "ris_noma/multik_benchmark.py", "ris_noma/multik_benchmark_analysis.py",
                 "ris_noma/stable_assessment.py", "ris_noma/multik_benchmark_cache.py")
FIELDS = {"scope", "calibration_layout_ids", "benchmark_layout_ids", "public_counts", "power_dbm", "xi",
    "search_margins", "training_seeds", "pso_seeds", "search_samples", "selection_samples", "audit_samples",
    "full_evaluations", "short_evaluations", "alpha", "carrier_hz", "spacing_wavelengths", "array_axis",
    "calibration_profiles", "benchmark_profiles", "calibration_deployments", "benchmark_deployments",
    "calibration_xi", "calibration_margin", "sic_rate_margin", "bootstrap_repetitions"}


def validate_settings(settings):
    c = deepcopy(settings)
    if set(c) != FIELDS or c["scope"] not in ("development_pilot", "smoke"):
        raise ValueError("Unexpected 7d fields/scope")
    for key in ("calibration_layout_ids", "benchmark_layout_ids", "public_counts", "training_seeds", "pso_seeds"):
        values = c[key]
        if not isinstance(values, list) or not values or values != sorted(set(values)) or any(type(v) is not int or v < 0 for v in values):
            raise ValueError(f"{key} must contain sorted distinct nonnegative integers")
    if min(c["public_counts"]) < 1 or min(c["calibration_layout_ids"] + c["benchmark_layout_ids"]) < 1000:
        raise ValueError("Positive K and new layout IDs >=1000 required")
    if set(c["calibration_layout_ids"]) & set(c["benchmark_layout_ids"]):
        raise ValueError("Calibration and benchmark layouts must be separate")
    if len(c["training_seeds"]) != len(c["pso_seeds"]):
        raise ValueError("Declare one paired PSO seed per training-seed replicate")
    for key in ("power_dbm", "xi", "calibration_xi", "search_margins"):
        a = c[key]
        if not isinstance(a, list) or not a or a != sorted(set(a)) or any(type(v) not in (int, float) or not np.isfinite(v) for v in a):
            raise ValueError(f"{key} must contain sorted distinct finite numbers")
    margins = [*c["search_margins"], c["calibration_margin"]]
    if any(not np.isfinite(v) or v < 0 for v in margins) or any(not 0 < xi < 1 for xi in c["xi"] + c["calibration_xi"]):
        raise ValueError("Invalid DEP targets/margins")
    if max(c["xi"] + c["calibration_xi"]) + max(margins) >= 1:
        raise ValueError("Every DEP plus search margin must be strictly below one")
    if not set(c["calibration_xi"]) <= set(c["xi"]):
        raise ValueError("Calibration DEP values must be a declared subset")
    for key in ("search_samples", "selection_samples", "audit_samples"):
        if type(c[key]) is not int or c[key] < 1000:
            raise ValueError("At least 1000 samples per bank required")
    for key in ("full_evaluations", "short_evaluations"):
        if type(c[key]) is not int or c[key] < 30 or c[key] % 30:
            raise ValueError("PSO budgets include initialization and must be multiples of 30")
    if c["short_evaluations"] >= c["full_evaluations"]:
        raise ValueError("Full PSO must have more evaluations than short PSO")
    if not np.isfinite(c["alpha"]) or not 0 < c["alpha"] < 1 or not np.isfinite(c["sic_rate_margin"]) or c["sic_rate_margin"] < 0:
        raise ValueError("Invalid confidence/SIC rate margin")
    if type(c["bootstrap_repetitions"]) is not int or c["bootstrap_repetitions"] < 100:
        raise ValueError("Use at least 100 clustered bootstrap replicates")
    for key in ("calibration_profiles", "benchmark_profiles"):
        if not c[key] or len(c[key]) != len(set(c[key])) or not set(c[key]) <= set(PROFILES):
            raise ValueError("Unknown/duplicate profiles")
    if set(c["benchmark_profiles"]) - {"legacy", "geometry"}:
        raise ValueError("Frozen 7c benchmark is limited to legacy and LoS-transfer profiles; new link budgets require new training before fair learned-solver claims")
    for key in ("calibration_deployments", "benchmark_deployments"):
        ids = []
        for deployment in c[key]:
            if not {"id", "alice", "ris"} <= set(deployment) or set(deployment) - {"id", "alice", "ris", "array_axis"}:
                raise ValueError("Deployment needs id, Alice and RIS coordinates")
            if not isinstance(deployment["id"], str) or not re.fullmatch(r"[a-z][a-z0-9_]*", deployment["id"]):
                raise ValueError("Unsafe deployment ID")
            if any(np.asarray(deployment[name], float).shape != (2,) or not np.all(np.isfinite(deployment[name])) for name in ("alice", "ris")):
                raise ValueError("Invalid deployment coordinates")
            if np.array_equal(deployment["alice"], deployment["ris"]):
                raise ValueError("Alice/RIS cannot coincide")
            array_elements(deployment["ris"], 32, c["carrier_hz"], c["spacing_wavelengths"], deployment.get("array_axis", c["array_axis"]))
            ids.append(deployment["id"])
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Use nonempty distinct deployment IDs")
    return c


def make_jobs(c, seen_ks):
    jobs = []
    for stage in ("calibration", "benchmark"):
        xis = c["calibration_xi"] if stage == "calibration" else c["xi"]
        profiles = c[stage + "_profiles"]; deployments = c[stage + "_deployments"]
        margins = [c["calibration_margin"]] if stage == "calibration" else c["search_margins"]
        replicates = 1 if stage == "calibration" else len(c["training_seeds"])
        cases = len(xis) * len(c["power_dbm"]) * len(margins) * replicates
        count = cases * (1 if stage == "calibration" else len(METHODS))
        family = cases * (5 if stage == "calibration" else 5 * 5 + 3)
        evaluations = cases * (c["full_evaluations"] if stage == "calibration" else c["full_evaluations"] + 4 * c["short_evaluations"])
        for profile, deployment, lid, k in itertools.product(profiles, deployments, c[stage + "_layout_ids"], c["public_counts"]):
            jobs.append({"id": f"{stage}_{profile}_{deployment['id']}_l{lid:06d}_k{k}",
                "stage": stage, "profile": profile, "deployment": deployment,
                "layout_id": lid, "n_public": k, "split": "development",
                "regime": "seen_k" if k in seen_ks else "unseen_k",
                "planned_search_pools": count, "selection_family": family,
                "audit_family": count * len(SELECTORS), "planned_objective_evaluations": evaluations})
    return jobs


def verify_manifest(root, manifest):
    root = Path(root).resolve()
    for name, digest in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Changed or missing tracked file: {name}")


def inspect(dataset, training, settings, workers=1, search_cache=None):
    from .multik_learning import check as check_training
    from .multik_graph import fit_normalizer
    import torch
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use 1 to 4 workers; 2 recommended for 100k banks")
    dataset, training = Path(dataset).resolve(), Path(training).resolve()
    c = validate_settings(settings)
    verified = check_training(training, dataset)
    data_plan = read_json(dataset / "plan.json"); source = read_json(dataset / "source_physics_plan.json")
    old = read_json(training / "plan.json")
    verify_manifest(dataset, read_json(dataset / "completion.json")["sha256"])
    verify_manifest(ROOT, old["code_sha256"])
    old_ids = {lid for values in data_plan["config"]["splits"].values() for lid in values}
    ids = set(c["calibration_layout_ids"] + c["benchmark_layout_ids"])
    if old_ids & ids:
        raise ValueError("Use development layouts outside all source 7b splits")
    if not set(c["power_dbm"]) <= set(data_plan["config"]["power_dbm"]) or not set(c["xi"]) <= set(data_plan["config"]["xi"]):
        raise ValueError("Frozen benchmark powers/DEP must have train lookup entries")
    selection = read_json(training / "selection_frozen.json")
    models = {f"{kind}_s{seed}": selection["views"][f"{kind}_s{seed}_physical"] for kind in ("gnn", "mlp") for seed in c["training_seeds"]}
    jobs = make_jobs(c, old["training_public_counts"])
    for job in jobs:
        adapted = deepcopy(source); adapted["base_config"]["positions"].update(alice=job["deployment"]["alice"], ris=job["deployment"]["ris"])
        # Invalid geometries are an explicit setup error, never silently
        # redrawn or removed from an evaluation denominator.
        physical, layout = geometry(data_plan, adapted, job, c["search_samples"])
        array_elements(physical["positions"]["ris"], physical["n_ris"], c["carrier_hz"], c["spacing_wavelengths"], job["deployment"].get("array_axis", c["array_axis"]))
        for power in c["power_dbm"]:
            from .config import validate_config
            validate_config(dict(physical, total_dbm=power))
    train = read_jsonl(dataset / "graphs/train.jsonl")
    normalizer = fit_normalizer([r["graph"] for r in train])
    if normalizer != old["normalization"]:
        raise ValueError("Frozen model normalization differs from train-only source")
    policy = table_policy(train, source, c["sic_rate_margin"])
    files = tuple(dict.fromkeys((*old["code_sha256"], *NUMERIC_FILES)))
    result = {"step": "7d", "format": "geometry_transfer_and_paired_pso_7d_v1", "config": c,
        "jobs": jobs, "methods": list(METHODS), "selectors": list(SELECTORS), "models": models,
        "normalization": normalizer, "lookup_policy": policy, "pso_settings": data_plan["pso_settings"],
        "streams": STREAMS, "workers": workers, "threads_per_worker": 1,
        "code_sha256": {name: sha256(ROOT / name) for name in files},
        "dataset_sha256": old["dataset_sha256"], "training_completion_sha256": sha256(training / "completion.json"),
        "source_7c_verified_files": verified["verified_files"],
        "planned_search_pools": sum(j["planned_search_pools"] for j in jobs),
        "planned_audit_rows": sum(j["audit_family"] for j in jobs),
        "planned_objective_evaluations": sum(j["planned_objective_evaluations"] for j in jobs),
        "versions": {"system": platform.system(), "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "torch": str(torch.__version__)},
        "input_policy": "7c physical selections for every declared training seed; no retraining or test-based model/lookup fitting",
        "profile_policy": "benchmark legacy vs geometry isolates LoS phase; geometry_friis only development calibration; no claim frozen networks were trained in new physical regime",
        "selection_policy": "same five-slot search archive for every PSO method, protected exact warm start; select on independent bank by feasibility then Bob rate",
        "confidence_policy": "alpha/(2*all_jobs) at selection/audit; declared finite family includes every method, seed, margin, P and DEP; repeated views count in family",
        "timing_policy": "cached-bank return time includes graph/forward, objective setup, search and online candidate scoring; also report full bank generation separately; final audit excluded from online latency",
        "random_policy": "common normalized NLoS and PSO draws across profiles/margins/methods; independent streams between search, selection, audit",
        "infrastructure_policy": "Alice is coordinate origin; shifted relative RIS deployment appears in calibration; benchmark deployment groups explicitly configured",
        "final_unseen_test": False, "new_weights_trained": False, "search_origin": None,
        "arithmetic_policy": "unchanged physical metric arithmetic; Willie power-identity assertion bounded by 32*machine_epsilon*power_scale; no tolerance added to feasibility thresholds"}
    if search_cache is not None:
        from .multik_benchmark_cache import validate_cache
        result["search_origin"] = validate_cache(search_cache, result, ROOT)
    return result


def stage_paths(output, phase, jobs):
    return [Path(output) / phase / (job["id"] + ".json") for job in jobs]


def freeze(output, phase, jobs):
    paths = stage_paths(output, phase, jobs)
    if any(read_group(output, phase, job) is None for job in jobs):
        raise ValueError("Cannot freeze incomplete stage")
    immutable(Path(output) / f"{phase}_frozen.json", {"phase": phase,
        "plan_sha256": sha256(Path(output) / "plan.json"),
        "sha256": {path.relative_to(output).as_posix(): sha256(path) for path in paths}})


def execute_stage(output, phase, jobs, workers, training=None):
    pending = [j for j in jobs if read_committed(Path(output) / phase / (j["id"] + ".json")) is None]
    done = len(jobs) - len(pending)
    print(f"7d {phase}: {done}/{len(jobs)} groups committed", flush=True)
    function = {"search": search_job, "selection": select_job, "audit": audit_job}[phase]
    arguments = [(str(output), j, str(training)) if phase == "search" else (str(output), j) for j in pending]
    if workers == 1:
        for arg in arguments:
            name = function(arg); done += 1
            print(f"7d {phase} {done}/{len(jobs)}: {name}", flush=True)
    elif arguments:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            futures = [pool.submit(function, arg) for arg in arguments]
            for future in as_completed(futures):
                name = future.result(); done += 1
                print(f"7d {phase} {done}/{len(jobs)}: {name}", flush=True)
    freeze(output, phase, jobs)


def run(dataset, training, output, settings, workers=1, resume=False, search_cache=None):
    output = Path(output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock", timeout=0):
        plan = inspect(dataset, training, settings, workers, search_cache)
        if output.exists() and any(output.iterdir()):
            if not resume:
                raise ValueError("Output is nonempty; use --resume with the same plan or a new output")
            if read_json(output / "plan.json") != plan:
                raise ValueError("Changed config, inputs, code, workers or runtime; use a new output")
            if (output / "completion.json").exists(): return check(output, dataset, training)
        output.mkdir(exist_ok=True)
        immutable(output / "plan.json", plan)
        for source_name, dest in (("source_physics_plan.json", "source_physics_plan.json"), ("plan.json", "source_dataset_plan.json")):
            immutable_bytes(output / dest, (Path(dataset) / source_name).read_bytes())
        immutable_bytes(output / "source_7c_selection.json", (Path(training) / "selection_frozen.json").read_bytes())
        for name in plan["code_sha256"]:
            immutable_bytes(output / "source_snapshot" / name, (ROOT / name).read_bytes())
        if search_cache is not None:
            from .multik_benchmark_cache import install_cache
            install_cache(search_cache, output, plan["search_origin"])
        execute_stage(output, "search", plan["jobs"], workers, training)
        verify_manifest(ROOT, plan["code_sha256"])
        execute_stage(output, "selection", plan["jobs"], workers)
        verify_manifest(ROOT, plan["code_sha256"])
        execute_stage(output, "audit", plan["jobs"], workers)
        verify_manifest(ROOT, plan["code_sha256"])
        from .multik_benchmark_analysis import summarize
        summary = summarize(output)
        verify_manifest(Path(dataset), plan["dataset_sha256"])
        if sha256(Path(training) / "completion.json") != plan["training_completion_sha256"]:
            raise ValueError("Source training run changed")
        for ref in plan["models"].values():
            if sha256(Path(training) / ref["path"]) != ref["sha256"]: raise ValueError("Model changed")
        paths = [p for p in sorted(output.rglob("*")) if p.is_file() and p.name != "completion.json" and "figures" not in p.relative_to(output).parts and not p.name.endswith(".tmp")]
        immutable(output / "completion.json", {"sha256": {p.relative_to(output).as_posix(): sha256(p) for p in paths}})
        return check(output, dataset, training)


def check(output, dataset=None, training=None):
    output = Path(output).resolve()
    completion = read_json(output / "completion.json")["sha256"]
    verify_manifest(output, completion)
    plan = read_json(output / "plan.json"); validate_settings(plan["config"])
    verify_manifest(ROOT, plan["code_sha256"])
    from .multik_benchmark_cache import verify_origin
    verify_origin(output, plan, ROOT)
    verify_manifest(output / "source_snapshot", plan["code_sha256"])
    if plan["final_unseen_test"] or plan["new_weights_trained"]: raise ValueError("Unexpected 7d scope")
    source = read_json(output / "source_dataset_plan.json")
    expected_jobs = make_jobs(plan["config"], source["config"]["training_public_counts"])
    if expected_jobs != plan["jobs"]: raise ValueError("Jobs differ from declared factorial plan")
    pools_count = rows_count = evaluations = 0
    for phase in ("search", "selection", "audit"):
        frozen = read_json(output / f"{phase}_frozen.json")
        expected = {path.relative_to(output).as_posix() for path in stage_paths(output, phase, plan["jobs"])}
        if set(frozen["sha256"]) != expected or frozen["plan_sha256"] != sha256(output / "plan.json"):
            raise ValueError("Frozen stage coverage/anchor differs")
        if any(completion.get(name) != digest for name, digest in frozen["sha256"].items()):
            raise ValueError("Stage hash differs from completion")
    for job in plan["jobs"]:
        records = {phase: read_group(output, phase, job) for phase in ("search", "selection", "audit")}
        search, selection, audit = (records[name] for name in ("search", "selection", "audit"))
        if not search["geometry_valid"] or len(search["pools"]) != job["planned_search_pools"]:
            raise ValueError("Missing/invalid geometry or planned pools; no denominator filtering is allowed")
        if search["objective_evaluations"] != job["planned_objective_evaluations"]:
            raise ValueError("PSO evaluation budget differs")
        pools_count += len(search["pools"]); evaluations += search["objective_evaluations"]
        config = plan["config"]
        calibration = job["stage"] == "calibration"
        xis = config["calibration_xi"] if calibration else config["xi"]
        margins = [config["calibration_margin"]] if calibration else config["search_margins"]
        replicates = [(None, config["pso_seeds"][0])] if calibration else list(zip(config["training_seeds"], config["pso_seeds"]))
        methods = ["full_pso"] if calibration else METHODS
        expected_axes = {(power, xi, margin, train_seed, pso_seed, method)
            for power, xi, margin, (train_seed, pso_seed), method in itertools.product(config["power_dbm"], xis, margins, replicates, methods)}
        actual_axes = [(pool["power_dbm"], pool["xi"], pool["search_margin"], pool["training_seed"], pool["pso_seed"], pool["method"]) for pool in search["pools"]]
        if len(set(actual_axes)) != len(actual_axes) or set(actual_axes) != expected_axes:
            raise ValueError("Missing or duplicated method/seed/power/DEP/margin combinations")
        if selection["search_frozen_sha256"] != sha256(output / "search_frozen.json") or audit["selection_frozen_sha256"] != sha256(output / "selection_frozen.json"):
            raise ValueError("Search/selection/audit chain changed")
        if len(selection["rows"]) != job["audit_family"] or len(audit["rows"]) != len(selection["rows"]):
            raise ValueError("Wrong final candidate count")
        for phase, record in records.items():
            meta = record["metadata"]
            if meta["bank_role"] != phase or meta["bank_samples"] != plan["config"][{"search":"search_samples","selection":"selection_samples","audit":"audit_samples"}[phase]]:
                raise ValueError("Bank role/sample count differs")
            if meta["los_sha256"] != search["metadata"]["los_sha256"] or meta["receiver_xy"] != search["metadata"]["receiver_xy"]:
                raise ValueError("Geometry/LoS changed between banks of one case")
            if meta["bank_entropy"][-1] != STREAMS[phase]: raise ValueError("Bank role seed mismatch")
        for index, pool in enumerate(search["pools"]):
            is_pso = pool["method"].endswith("pso")
            expected_budget = (plan["config"]["full_evaluations"] if pool["method"] == "full_pso" else plan["config"]["short_evaluations"]) if is_pso else 0
            if pool["objective_evaluations"] != expected_budget or len(pool["candidates"]) > (5 if is_pso else 1):
                raise ValueError("Method-specific evaluation/archive budget differs")
            if is_pso and pool.get("warm_start_injected"):
                protected = next((candidate for candidate in pool["candidates"] if candidate["evaluation"] == 1), None)
                if protected is None or not np.allclose(protected["allocation"], pool["warm_start_allocation"], atol=2e-6, rtol=0):
                    raise ValueError("The exact warm-start candidate was not preserved")
            kind = pool["method"].removesuffix("_short_pso")
            if kind in ("gnn", "mlp") and pool.get("model_checkpoint_sha256") != plan["models"][f"{kind}_s{pool['training_seed']}"]["sha256"]:
                raise ValueError("A method used another model checkpoint")
            reviews = selection["reviewed_pools"][index]
            if reviews["pool_index"] != index or len(reviews["candidates"]) != len(pool["candidates"]): raise ValueError("Incomplete independent pool scoring")
            for candidate, scored in zip(pool["candidates"], reviews["candidates"]):
                if any(scored[key] != value for key, value in candidate.items()): raise ValueError("Selection altered a search proposal")
                if any(scored[key] != value for key, value in at_target(scored["physical"], pool["xi"]).items()): raise ValueError("Invalid pool feasibility flag")
                if scored["physical"]["constants"]["pool_size"] != job["selection_family"]: raise ValueError("Selection family differs")
            valid = [r for r in reviews["candidates"] if r["passed"]]
            chosen = min(valid, key=lambda r:(-r["physical"]["bob_rate"],r["candidate_id"])) if valid else None
            for selector in SELECTORS:
                row = selection["rows"][index*2 + SELECTORS.index(selector)]
                expected_id = pool["raw_candidate_id"] if selector == "search_best" else (None if chosen is None else chosen["candidate_id"])
                if row["pool_index"] != index or row["selector"] != selector or row["candidate_id"] != expected_id: raise ValueError("Final choice differs from declared selector")
                expected_candidate = next((candidate for candidate in reviews["candidates"] if candidate["candidate_id"] == expected_id), None)
                if row["returned"] != (expected_candidate is not None) or row["allocation"] != (None if expected_candidate is None else expected_candidate["allocation"]):
                    raise ValueError("Returned allocation differs from selected candidate")
        for before, row in zip(selection["rows"], audit["rows"]):
            if any(row[key] != value for key, value in before.items()): raise ValueError("Final audit modified a selected proposal")
            if row["returned"]:
                if any(row[key] != value for key, value in at_target(row,row["xi"]).items()): raise ValueError("Final feasibility decision differs")
                if row["constants"]["pool_size"] != job["audit_family"]: raise ValueError("Audit family differs")
            elif row["passed"] or row["gated_bob_rate"] != 0.: raise ValueError("A missing solution cannot pass")
        rows_count += len(audit["rows"])
    if (pools_count, rows_count, evaluations) != (plan["planned_search_pools"], plan["planned_audit_rows"], plan["planned_objective_evaluations"]):
        raise ValueError("Global factorial budget differs")
    from .multik_benchmark_analysis import summarize
    expected = summarize(output, write=False)
    actual = read_json(output / "summary.json")
    if expected != actual: raise ValueError("Summary differs from raw audits")
    if dataset is not None: verify_manifest(Path(dataset), plan["dataset_sha256"])
    if training is not None:
        if sha256(Path(training) / "completion.json") != plan["training_completion_sha256"]: raise ValueError("Training run changed")
        for ref in plan["models"].values():
            if sha256(Path(training) / ref["path"]) != ref["sha256"]: raise ValueError("Frozen model changed")
    return actual | {"verified_files": len(completion)}


def status(output):
    output = Path(output)
    plan = read_json(output / "plan.json")
    return {"step": "7d", "complete": (output / "completion.json").exists(), "groups": len(plan["jobs"]),
        **{phase: sum(read_committed(path) is not None for path in stage_paths(output, phase, plan["jobs"])) for phase in ("search", "selection", "audit")}}
