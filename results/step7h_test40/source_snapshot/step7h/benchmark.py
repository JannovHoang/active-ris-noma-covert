"""Paired fresh PSO searches, globally frozen outputs, independent physical audit."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
from pathlib import Path
import multiprocessing as mp
import time
import numpy as np

from .common import (METHODS, SHORT, DIRECT, read, write, digest, commit, committed, freeze, verify, safe_path)
from .physics import context, corrected, pso_rng
from ris_noma.channels import public_order
from ris_noma.continuous_data import MarginObjective
from ris_noma.curve_models import load_model
from ris_noma.learning_train import configure_runtime
from ris_noma.multik_core import decode_allocation, original_allocation
from ris_noma.multik_graph import statistical_graph
from ris_noma.multik_models import predict
from ris_noma.pso import PSOSettings, optimize
from ris_noma.stable_assessment import raw_assessment
from ris_noma.multik_data_core import at_target


def variants(plan):
    result = [("full_pso", plan["settings"]["full_budget"]), *( (m, 0) for m in DIRECT)]
    return result + [(method, budget) for budget in plan["short_budgets"] for method in SHORT]


def expected_keys(plan, job):
    return {(case["case_index"], r["training_seed"], method, budget)
            for case in job["cases"] for r in plan["settings"]["replicates"] for method, budget in variants(plan)}


def key(row):
    return row["case_index"], row["replicate"], row["method"], row["budget"]


def row_base(plan, job, case, replicate, method, budget):
    return {"job_id": job["id"], "case_id": f"{job['id']}_c{case['case_index']:03d}",
            "case_index": case["case_index"], "layout_id": job["layout_id"], "n_public": job["n_public"],
            "power_dbm": case["power_dbm"], "xi": case["xi"], "regime": job["regime"],
            "replicate": replicate["training_seed"], "pso_seed": replicate["pso_seed"],
            "training_seed": replicate["training_seed"] if method.startswith(("gnn", "mlp")) else None,
            "method": method, "budget": budget, "phase": plan["stage"]}


def run_search(objective, xi, k, settings, rng, initial=None):
    counter = 0
    def evaluate(z):
        nonlocal counter
        counter += 1
        return objective.evaluate(z, xi)
    result = optimize(evaluate, k+1, settings, rng, initial_positions=initial)
    if counter != settings.swarm_size * settings.sweeps or result.evaluations != counter:
        raise ArithmeticError("Actual objective-call count differs from the fixed PSO budget")
    return result


def empty_search(plan, job, error):
    rows = []
    for case in job["cases"]:
        for rep in plan["settings"]["replicates"]:
            for method, budget in variants(plan):
                rows.append({**row_base(plan, job, case, rep, method, budget),
                    "allocation": None, "returned": False, "objective_evaluations": 0,
                    "search_feasible": False, "search_rate": None, "search_dep_gamma": None,
                    "order": None, "feature_cv_willie": None, "cv_stratum": "geometry_rejected",
                    "history": [], "input_bank_amortized_seconds": 0., "graph_seconds": 0.,
                    "forward_seconds": 0., "correction_seconds": 0., "objective_setup_seconds": 0.,
                    "pso_seconds": 0., "online_seconds": 0., "geometry_error": error})
    return rows


def search_task(args):
    output, job = args
    output = Path(output)
    configure_runtime(0, 1, "cpu")
    plan = read(output / "plan.json")
    path = output / f"search/{job['id']}.json"
    old = committed(path)
    if old is not None:
        if old["plan_sha256"] != digest(output / "plan.json"): raise ValueError("Search belongs to another plan")
        return job["id"]
    role = {"development": "development_input", "final": "final_input", "timing": "timing_input"}[plan["stage"]]
    try:
        c, layout, bank, metadata = context(output, plan, job, role)
    except ValueError as error:
        if not str(error).startswith("Layout bị loại:"): raise
        commit(path, {"job": job, "rows": empty_search(plan, job, str(error)), "geometry_error": str(error),
                      "metadata": {}, "graphs": [], "plan_sha256": digest(output / "plan.json"), "objective_evaluations": 0})
        return job["id"]
    refs = read(output / "models.json")
    models = {}
    cold = {}
    for rep in plan["settings"]["replicates"]:
        for kind in ("gnn", "mlp"):
            name = f"{kind}_s{rep['training_seed']}"
            ref = refs[name]
            model_path = safe_path(output, ref["path"])
            if digest(model_path) != ref["sha256"]: raise ValueError("Checkpoint changed before search")
            started = time.perf_counter()
            network, saved = load_model(model_path)
            if saved["normalization"] != ref["normalization"]: raise ValueError("Normalizer metadata changed")
            models[name] = network
            cold[name] = time.perf_counter()-started
    order = public_order(bank)
    warm_graph = statistical_graph(dict(c, total_dbm=job["cases"][0]["power_dbm"]), layout, bank, order, job["cases"][0]["xi"])
    for name, model in models.items(): predict(model, [warm_graph], refs[name]["normalization"])
    rows, graphs, evaluations = [], [], 0
    for case in job["cases"]:
        radio = dict(c, total_dbm=case["power_dbm"])
        started = time.perf_counter()
        graph = statistical_graph(radio, layout, bank, order, case["xi"])
        graph_seconds = time.perf_counter()-started
        graphs.append({"case_index": case["case_index"], "graph": graph})
        cv = float(graph["global_raw"][18])
        started = time.perf_counter()
        objective = MarginObjective(radio, layout, bank, order, plan["settings"]["search_qos_margin"])
        objective_seconds = time.perf_counter()-started
        for rep in plan["settings"]["replicates"]:
            forecasts = {}
            for kind in ("gnn", "mlp"):
                name = f"{kind}_s{rep['training_seed']}"
                started = time.perf_counter()
                raw = predict(models[name], [graph], refs[name]["normalization"])[0]
                forward = time.perf_counter()-started
                started = time.perf_counter()
                y = corrected(raw, refs[name]["delta"], radio["bounds"])
                forecasts[kind] = {"allocation": y, "raw_allocation": raw.tolist(), "forward": forward,
                    "correction": time.perf_counter()-started, "delta": refs[name]["delta"],
                    "checkpoint_sha256": refs[name]["sha256"], "epoch": refs[name]["epoch"]}
            choices = variants(plan)
            rotation = (job["layout_id"]+case["case_index"]+rep["pso_seed"]) % len(choices)
            for method, budget in choices[rotation:]+choices[:rotation]:
                kind = "gnn" if method.startswith("gnn") else "mlp" if method.startswith("mlp") else None
                forecast = forecasts.get(kind)
                row = row_base(plan, job, case, rep, method, budget)
                row.update(order=order.tolist(), returned=True, feature_cv_willie=cv,
                    cv_stratum="low" if cv < plan["settings"]["low_cv_threshold"] else "regular",
                    input_bank_amortized_seconds=metadata["bank_generation_seconds"]/len(job["cases"]),
                    graph_seconds=graph_seconds if kind else 0., forward_seconds=forecast["forward"] if kind else 0.,
                    correction_seconds=forecast["correction"] if kind else 0.,
                    objective_setup_seconds=objective_seconds if budget else 0.,
                    warm_particles=1 if method in ("gnn_short_pso", "mlp_short_pso") else 0,
                    delta=forecast["delta"] if kind else 0.,
                    checkpoint_sha256=forecast["checkpoint_sha256"] if kind else None,
                    epoch=forecast["epoch"] if kind else None)
                if budget:
                    initial = None
                    if method != "full_pso" and method != "random_short_pso":
                        y = forecast["allocation"]
                        _, center = original_allocation(y[0], y[1:-1][order], y[-1], order, radio["bounds"])
                        initial = center[None, :]
                    cfg = PSOSettings(**dict(plan["pso_settings"], swarm_size=30, sweeps=budget//30))
                    started = time.perf_counter()
                    result = run_search(objective, case["xi"]+plan["settings"]["search_dep_margin"],
                        job["n_public"], cfg, pso_rng(plan, c, job, rep["pso_seed"], case["case_index"]), initial)
                    elapsed = time.perf_counter()-started
                    mb, public, split = decode_allocation(result.position, job["n_public"], radio["bounds"])
                    y, _ = original_allocation(mb, public, split, order, radio["bounds"])
                    row.update(allocation=y.tolist(), pso_seconds=elapsed,
                        objective_evaluations=result.evaluations, search_feasible=bool(result.outcome.feasible),
                        search_rate=result.outcome.score, search_dep_gamma=result.outcome.metrics["dep_min"],
                        search_violation=result.outcome.violation, history=result.history)
                    evaluations += result.evaluations
                else:
                    row.update(allocation=forecast["allocation"].tolist(), pso_seconds=0., objective_evaluations=0,
                               search_feasible=None, search_rate=None, search_dep_gamma=None, search_violation=None, history=[])
                row["online_seconds"] = sum(row[n] for n in ("input_bank_amortized_seconds", "graph_seconds", "forward_seconds",
                    "correction_seconds", "objective_setup_seconds", "pso_seconds"))
                rows.append(row)
    if {key(r) for r in rows} != expected_keys(plan, job) or len(rows) != len(expected_keys(plan, job)):
        raise ArithmeticError("Incomplete or duplicated search product")
    commit(path, {"job": job, "rows": rows, "graphs": graphs, "metadata": metadata,
        "cold_model_load_seconds": cold, "model_warmup_excluded_from_warm_return_time": True,
        "timing_policy": "Bank generation amortized over eight (or smoke) cases, not over methods or seed replicates; each method charged its standalone feature/forward cost; cold model load separate",
        "objective_evaluations": evaluations, "plan_sha256": digest(output / "plan.json")})
    return job["id"]


def audit_task(args):
    output, job = args
    output = Path(output)
    configure_runtime(0, 1, "cpu")
    plan = read(output / "plan.json")
    path = output / f"audit/{job['id']}.json"
    old = committed(path)
    anchor_sha = digest(output / "allocations_frozen.json")
    if old is not None:
        if old["allocations_sha256"] != anchor_sha: raise ValueError("Audit belongs to different frozen proposals")
        return job["id"]
    proposal = committed(output / f"search/{job['id']}.json")
    if proposal is None: raise ValueError("Search receipt is missing")
    if proposal.get("geometry_error"):
        rows = [{**r, "passed": False, "point_passed": False, "invalid_layout_assumption": True,
                 "bob_rate": 0., "gated_bob_rate": 0., "dep_gamma": None, "dep_empirical": None,
                 "dep_lower": None, "dep_lower_margin": None, "qos_lower_margin": None,
                 "failed_checks": ["geometry_rejected"]} for r in proposal["rows"]]
        metadata, elapsed, unique = {}, 0., 0
    else:
        role = {"development": "development_audit", "final": "final_audit", "timing": "timing_audit"}[plan["stage"]]
        c, layout, bank, metadata = context(output, plan, job, role)
        cache, rows = {}, []
        started = time.perf_counter()
        for row in proposal["rows"]:
            k = (row["power_dbm"], tuple(row["allocation"]))
            if k not in cache:
                cache[k] = raw_assessment(dict(c, total_dbm=row["power_dbm"]), layout, bank,
                    np.asarray(row["order"], int), row["allocation"], plan["certificate_pool_size"], plan["certificate_alpha_per_group"])
            raw = cache[k]
            decision = at_target(raw, row["xi"])
            rows.append({**{k:v for k,v in row.items() if k != "history"}, **raw, **decision,
                "point_passed": bool(raw["dep_gamma"] >= row["xi"] and all(raw["point_checks_without_dep"].values())),
                "invalid_layout_assumption": not raw["point_checks_without_dep"]["layout_bob_mean_gain_rule"],
                "dep_lower_margin": raw["dep_lower"]-row["xi"]})
        elapsed, unique = time.perf_counter()-started, len(cache)
    commit(path, {"job": job, "rows": rows, "metadata": metadata, "scoring_seconds": elapsed,
                  "unique_allocation_evaluations": unique, "allocations_sha256": anchor_sha,
                  "plan_sha256": digest(output / "plan.json")})
    return job["id"]


def execute_jobs(function, args, workers, label):
    if not args: return
    if workers == 1:
        for i, arg in enumerate(args, 1):
            name = function(arg)
            print(f"7h {label} {i}/{len(args)}: {name}", flush=True)
    else:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
        iterator, pending, finished = iter(args), set(), 0
        try:
            for _ in range(min(workers,len(args))): pending.add(pool.submit(function,next(iterator)))
            while pending:
                done, pending = wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    name = future.result()
                    finished += 1
                    print(f"7h {label} {finished}/{len(args)}: {name}",flush=True)
                    arg = next(iterator,None)
                    if arg is not None: pending.add(pool.submit(function,arg))
        except BaseException:
            for future in pending: future.cancel()
            pool.shutdown(wait=False,cancel_futures=True)
            raise
        else: pool.shutdown(wait=True)


def freeze_search(output, plan):
    output = Path(output)
    paths = []
    for job in plan["jobs"]:
        path = output / f"search/{job['id']}.json"
        record = committed(path)
        if record is None or record["plan_sha256"] != digest(output / "plan.json"):
            raise ValueError("Every declared search must complete before audit")
        if len(record["rows"]) != len(expected_keys(plan, job)) or {key(r) for r in record["rows"]} != expected_keys(plan, job):
            raise ValueError("Search case/method/replicate product changed")
        paths.extend((path, Path(str(path)+".receipt.json")))
    return freeze(output, "allocations_frozen.json", paths, plan_sha256=digest(output / "plan.json"),
                  policy="All final allocations frozen globally before any audit fading; no audit fallback")


def run_stage(output, plan):
    output = Path(output)
    pending = [j for j in plan["jobs"] if committed(output / f"search/{j['id']}.json") is None]
    execute_jobs(search_task, [(str(output), j) for j in pending], plan["workers"], plan["stage"]+" search")
    freeze_search(output, plan)
    verify(output, read(output / "allocations_frozen.json"))
    pending = [j for j in plan["jobs"] if committed(output / f"audit/{j['id']}.json") is None]
    execute_jobs(audit_task, [(str(output), j) for j in pending], plan["workers"], plan["stage"]+" audit")
    paths = []
    for job in plan["jobs"]:
        path = output / f"audit/{job['id']}.json"
        record = committed(path)
        if record["allocations_sha256"] != digest(output / "allocations_frozen.json"):
            raise ValueError("Audit was not anchored to the final proposal family")
        paths.extend((path, Path(str(path)+".receipt.json")))
    freeze(output, "audit_frozen.json", paths, plan_sha256=digest(output / "plan.json"))


def all_rows(output, plan):
    return [r for job in plan["jobs"] for r in committed(Path(output)/f"audit/{job['id']}.json")["rows"]]
