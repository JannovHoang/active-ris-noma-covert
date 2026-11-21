"""7b dataset orchestration, grouped splits, immutable phases and export checks."""
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from copy import deepcopy
import json
import multiprocessing as mp
from pathlib import Path
import platform

from filelock import FileLock
import numpy as np
import scipy

from .config import validate_config
from .expanded_data import write_table
from .graph_data import NODE_FEATURES, GLOBAL_FEATURES, EDGE_FEATURES
from .learning_data import read_json, sha256
from .multik_core import extend_config
from .multik_data_core import (STREAMS, at_target, audit_group, choose_label,
                              context, raw_assessment, search_group, select_group)
from .multik_graph import SCHEMA_VERSION, validate_graph
from .pso import PSOSettings

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "multik_pso_graph_dataset_7b_v1"
PHASES = ("search", "selection", "heldout")
NUMERIC_FILES = ("run_step7b.py", "ris_noma/multik_data_core.py", "ris_noma/multik_dataset.py")


def write_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def write_json(path, value):
    write_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def immutable(path, value):
    if Path(path).exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen 7b file changed: {Path(path).name}")
    else:
        write_json(path, value)


def immutable_bytes(path, value):
    if Path(path).exists():
        if Path(path).read_bytes() != value:
            raise ValueError(f"Copied source bytes changed: {Path(path).name}")
    else:
        write_bytes(path, value)


def receipt_path(path):
    return Path(path).with_name(Path(path).name + ".receipt.json")


def read_committed(path):
    path = Path(path)
    receipt = receipt_path(path)
    if not receipt.exists():
        return None
    if not path.is_file() or read_json(receipt).get("sha256") != sha256(path):
        raise ValueError(f"Completed 7b group missing/changed: {path}")
    return read_json(path)


def validate_settings(settings):
    c = deepcopy(settings)
    expected = {"training_public_counts", "unseen_public_counts", "splits", "power_dbm", "xi", "search_margin",
                "search_samples", "selection_samples", "heldout_samples", "pso_seeds", "pso_evaluations", "alpha"}
    if set(c) != expected:
        raise ValueError("Unexpected 7b config fields")
    for name in ("training_public_counts", "unseen_public_counts"):
        if not isinstance(c[name], list) or not c[name] or any(type(k) is not int or k < 1 for k in c[name]) or c[name] != sorted(set(c[name])):
            raise ValueError("Use sorted distinct positive public counts")
    if set(c["training_public_counts"]) & set(c["unseen_public_counts"]):
        raise ValueError("Unseen K cannot occur in training/validation")
    if not isinstance(c["splits"], dict) or set(c["splits"]) != {"train", "validation", "test"}:
        raise ValueError("Declare train, validation, test layout groups")
    if any(not isinstance(ids, list) or not ids or any(type(i) is not int or i < 1000 for i in ids)
           or ids != sorted(set(ids)) for ids in c["splits"].values()):
        raise ValueError("Use nonempty sorted distinct layout IDs >=1000")
    ids = [i for values in c["splits"].values() for i in values]
    if len(ids) != len(set(ids)):
        raise ValueError("A layout cannot cross splits")
    for name in ("power_dbm", "xi"):
        values = c[name]
        if (not isinstance(values, list) or not values or any(type(v) not in (int, float) or not np.isfinite(v) for v in values)
                or values != sorted(set(values))):
            raise ValueError("Use sorted distinct finite powers and DEP targets")
    if (type(c["search_margin"]) not in (int, float) or not np.isfinite(c["search_margin"])
            or not 0 <= c["search_margin"] < 1 or any(not 0 < x < 1 or x + c["search_margin"] >= 1 for x in c["xi"])):
        raise ValueError("DEP plus search margin must be strictly below one")
    seeds = c["pso_seeds"]
    if not isinstance(seeds, list) or not seeds or any(type(i) is not int or i < 0 for i in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("PSO seeds must be distinct nonnegative integers")
    for name in ("search_samples", "selection_samples", "heldout_samples"):
        if type(c[name]) is not int or c[name] < 1000:
            raise ValueError("Use at least 1000 fading samples per bank")
    if type(c["pso_evaluations"]) is not int or c["pso_evaluations"] < 30 or c["pso_evaluations"] % 30:
        raise ValueError("PSO budget must be a positive multiple of 30")
    if type(c["alpha"]) not in (int, float) or not np.isfinite(c["alpha"]) or not 0 < c["alpha"] < 1:
        raise ValueError("Require finite 0<alpha<1")
    return c


def role(job):
    return "test_unseen_k" if job["regime"] == "unseen_k" else job["split"]


def group_name(job):
    return f"layout_{job['layout_id']:06d}_k{job['n_public']}.json"


def make_jobs(c):
    return [{"layout_id": lid, "n_public": k, "split": split,
             "regime": "seen_k" if k in c["training_public_counts"] else "unseen_k"}
            for split in ("train", "validation", "test") for lid in c["splits"][split]
            for k in sorted(c["training_public_counts"] + (c["unseen_public_counts"] if split == "test" else []))]


def inspect(previous, settings):
    # This import is parent-only; spawned data workers do not import Torch.
    from .multik_demo import check as check_7a
    previous = Path(previous).resolve()
    status = check_7a(previous)
    prior = read_json(previous / "plan.json")
    c = validate_settings(settings)
    old_ids = {i for values in prior["config"]["splits"].values() for i in values}
    if old_ids & {i for values in c["splits"].values() for i in values}:
        raise ValueError("Use new layout groups outside the 7a dataset")
    source = read_json(previous / "source_physics_plan.json")
    if source["base_config"]["los"]["mode"] != "explicit_phases":
        raise ValueError("7b preserves the explicit-phase LoS convention verified in 7a")
    for k in c["training_public_counts"] + c["unseen_public_counts"]:
        base = extend_config(source["base_config"], k)
        for power in c["power_dbm"]:
            validate_config(dict(base, total_dbm=power))
    for name, digest in prior["code_sha256"].items():
        if not (ROOT / name).is_file() or sha256(ROOT / name) != digest:
            raise ValueError(f"The verified 7a dependency changed: {name}")
    pso = dict(prior["pso_settings"], swarm_size=30, sweeps=c["pso_evaluations"] // 30)
    PSOSettings(**pso)
    jobs = make_jobs(c)
    cases_per_group = len(c["power_dbm"]) * len(c["xi"])
    counts = dict(Counter(role(job) for job in jobs))
    counts = {name: counts.get(name, 0) * cases_per_group for name in ("train", "validation", "test", "test_unseen_k")}
    tracked = tuple(dict.fromkeys((*prior["code_sha256"], *NUMERIC_FILES)))
    plan = {"step": "7b", "format": FORMAT, "graph_schema": SCHEMA_VERSION, "config": c,
            "jobs": jobs, "streams": STREAMS, "pso_settings": pso,
            "planned_counts": counts, "planned_cases": sum(counts.values()),
            "max_pso_evaluations": sum(counts.values()) * len(c["pso_seeds"]) * c["pso_evaluations"],
            "previous_sha256": {name: sha256(previous / name) for name in ("completion.json", "plan.json", "source_physics_plan.json", "summary.json")},
            "source_7a_complete": status["complete"],
            "code_sha256": {name: sha256(ROOT / name) for name in tracked},
            "versions": {"system": platform.system(), "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
            "source_los_policy": prior["los_extension"],
            "physics_policy": "unchanged original scalar Active-RIS noise, Bob phase alignment and average power budget",
            "geometry_policy": "same 7a uniform boxes; shared base layout split across K; no adaptive resampling; Alice/RIS fixed",
            "pool_policy": "one frozen candidate pool per layout/K/power shared across DEP; no cross-power reuse; no deduplication",
            "selection_policy": "search feasible at xi+margin AND independent selection passes at xi; highest SEARCH-bank Bob rate; tie by candidate_id",
            "heldout_policy": "record all selected labels and failures; never remove/reselect/repair using heldout",
            "confidence_policy": "alpha/(2*number_of_groups) per selection or heldout group; family includes all powers and candidates; uniform DEP bound covers every xi",
            "training_policy": "no training or normalization fitting in 7b; unseen-K labels restricted to test_unseen_k export",
            "runtime_policy": "spawn processes, one BLAS thread per worker; workers are not part of the scientific plan",
            "estimated_peak_bank_gib_per_worker": float(16 * max(c["selection_samples"], c["heldout_samples"])
                * source["base_config"]["n_ris"] * (4 * (max(c["training_public_counts"] + c["unseen_public_counts"]) + 2) + 4) / 1024**3)}
    return plan, source


def task(arguments):
    output_string, phase, job = arguments
    output = Path(output_string)
    path = output / phase / group_name(job)
    if read_committed(path) is not None:
        return group_name(job)
    plan, source = read_json(output / "plan.json"), read_json(output / "source_physics_plan.json")
    if phase == "search":
        result = search_group(plan, source, job)
    else:
        search = read_committed(output / "search" / group_name(job))
        if search is None:
            raise ValueError("Search group is not committed")
        if phase == "selection":
            result = select_group(plan, source, search, sha256(output / "search_frozen.json"))
        elif phase == "heldout":
            selected = read_committed(output / "selection" / group_name(job))
            if selected is None:
                raise ValueError("Selection group is not committed")
            result = audit_group(plan, source, search, selected, sha256(output / "selection_frozen.json"))
        else:
            raise ValueError("Unknown 7b phase")
    write_json(path, result)
    write_json(receipt_path(path), {"sha256": sha256(path)})
    return group_name(job)


def run_phase(output, phase, jobs, workers):
    pending = []
    for job in jobs:
        if read_committed(output / phase / group_name(job)) is None:
            pending.append(job)
    done_count = len(jobs) - len(pending)
    print(f"7b {phase}: {done_count}/{len(jobs)} groups already committed", flush=True)
    if workers == 1:
        for job in pending:
            name = task((str(output), phase, job))
            done_count += 1
            print(f"7b {phase} {done_count}/{len(jobs)}: {name}", flush=True)
    elif pending:
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
        active, queue = {}, iter(pending)
        def submit_one():
            job = next(queue, None)
            if job is not None:
                future = executor.submit(task, (str(output), phase, job))
                active[future] = job
        try:
            for _ in range(workers):
                submit_one()
            while active:
                finished, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in finished:
                    name = future.result()
                    del active[future]
                    done_count += 1
                    print(f"7b {phase} {done_count}/{len(jobs)}: {name}", flush=True)
                    submit_one()
        finally:
            for future in active:
                future.cancel()
            # At most `workers` groups are in flight. Keep the output lock until
            # they finish/exit so a new resume cannot race their file writes.
            executor.shutdown(wait=True, cancel_futures=True)
    paths = [output / phase / group_name(job) for job in jobs]
    for path in paths:
        if read_committed(path) is None:
            raise ValueError("Cannot freeze an incomplete data phase")
    immutable(output / f"{phase}_frozen.json", {"phase": phase,
              "sha256": {p.relative_to(output).as_posix(): sha256(p) for p in paths}})


def write_jsonl(path, rows):
    write_bytes(path, "".join(json.dumps(r, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n" for r in rows).encode("utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def summaries(plan, records):
    def count(rows):
        labeled = [r for r in rows if r["label_available"]]
        return {"cases": len(rows), "labels": len(labeled), "missing_labels": len(rows) - len(labeled),
                "heldout_passes": sum(r["heldout"]["passed"] for r in labeled),
                "heldout_failures_retained": sum(not r["heldout"]["passed"] for r in labeled)}
    counts = {name: count([r for r in records if role(r) == name]) for name in ("train", "validation", "test", "test_unseen_k")}
    coverage = [{"role": name, "n_public": k, "power_dbm": p, "xi": xi,
                 **count([r for r in records if (role(r), r["n_public"], r["power_dbm"], r["xi"]) == (name, k, p, xi)])}
                for name in counts for k in sorted(plan["config"]["training_public_counts"] + plan["config"]["unseen_public_counts"])
                for p in plan["config"]["power_dbm"] for xi in plan["config"]["xi"]
                if any((role(r), r["n_public"], r["power_dbm"], r["xi"]) == (name, k, p, xi) for r in records)]
    ready = all(any(role(r) == name and r["n_public"] == k and r["label_available"] for r in records)
                for name in ("train", "validation", "test") for k in plan["config"]["training_public_counts"])
    return counts, coverage, ready


def export_dataset(output, plan):
    records = [r for job in plan["jobs"] for r in read_committed(output / "heldout" / group_name(job))["rows"]]
    write_jsonl(output / "records.jsonl", records)
    for name in ("train", "validation", "test", "test_unseen_k"):
        subset = [{key: r[key] for key in ("layout_id", "n_public", "split", "regime", "power_dbm", "xi", "graph", "allocation", "order", "label_available", "heldout_used_for_label_filter")}
                  | {"heldout_passed": r["heldout"]["passed"]} for r in records if role(r) == name and r["label_available"]]
        write_jsonl(output / "graphs" / f"{name}.jsonl", subset)
    write_json(output / "feature_schema.json", {"version": SCHEMA_VERSION, "node_features": NODE_FEATURES,
               "global_features": GLOBAL_FEATURES, "edge_features": EDGE_FEATURES,
               "node_order": "Bob, public_original_0,...,public_original_K-1",
               "target_order": "mb, public_original_0,...,public_original_K-1, s",
               "ragged_graphs": True, "padding": "batch-time only; mask required",
               "normalizer": "fit in 7c on train only", "audit_fields_are_not_features": True})
    counts, coverage, ready = summaries(plan, records)
    write_table(output / "coverage.csv", coverage)
    fields = ("layout_id", "n_public", "split", "regime", "power_dbm", "xi", "geometry_valid", "layout_eligible", "label_available", "no_label_reason", "selected_candidate_id", "selected_origin_xi", "search_rate")
    write_table(output / "cases.csv", [{name: r[name] for name in fields} | {
                "heldout_passed": None if r["heldout"] is None else r["heldout"]["passed"],
                "heldout_failed_checks": None if r["heldout"] is None else r["heldout"]["failed_checks"]} for r in records])
    searches = [read_committed(output / "search" / group_name(job)) for job in plan["jobs"]]
    write_table(output / "layouts.csv", [{name: group[name] for name in ("layout_id", "n_public", "split", "regime", "geometry_valid", "geometry_error", "layout_eligible", "order", "positions")} for group in searches])
    result = {"step": "7b", "format": FORMAT, "complete": True, "training_ready_minimum_only": ready,
              "planned_cases": plan["planned_cases"], "base_layouts": sum(map(len, plan["config"]["splits"].values())),
              "layout_k_groups": len(plan["jobs"]), "counts": counts,
              "pso_evaluations": sum(g["pso_evaluations"] for g in searches),
              "pool_target_checks_outside_pso_budget": sum(g["pool_target_checks"] for g in searches),
              "ineligible_layout_k_groups": sum(g["geometry_valid"] and not g["layout_eligible"] for g in searches),
              "invalid_geometry_groups": sum(not g["geometry_valid"] for g in searches),
              "missing_label_reasons": dict(sorted(Counter(r["no_label_reason"] for r in records if not r["label_available"]).items())),
              "heldout_filtered_labels": False, "normalization_fitted": False, "gnn_trained": False,
              "unseen_k_in_train_or_validation": False,
              "scope": "expanded development dataset under fixed Alice/RIS and explicit LoS; no GNN superiority claim"}
    write_json(output / "summary.json", result)
    files = [p for p in sorted(output.rglob("*")) if p.is_file() and p.name != "completion.json"
             and not p.name.endswith(".tmp") and "figures" not in p.relative_to(output).parts]
    immutable(output / "completion.json", {"sha256": {p.relative_to(output).as_posix(): sha256(p) for p in files}})
    return result


def run(previous, output, settings, workers=1, resume=False):
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("Use 1..8 worker processes; start with 4 on a 32GB laptop")
    previous, output = Path(previous).resolve(), Path(output).resolve()
    if previous == output or previous in output.parents or output in previous.parents:
        raise ValueError("Use a separate non-nested dataset output")
    plan, source = inspect(previous, settings)
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output.with_name(f".{output.name}.lock")), timeout=0):
        if output.exists() and any(output.iterdir()):
            if not resume or not (output / "plan.json").is_file() or read_json(output / "plan.json") != plan:
                raise ValueError("Nonempty/changed 7b output: use --resume with the same inputs/code/config/environment or a new output")
            if (output / "completion.json").exists():
                return check(output)
            # Validate completed work BEFORE writing another group.
            for phase in PHASES:
                for job in plan["jobs"]:
                    read_committed(output / phase / group_name(job))
        output.mkdir(parents=True, exist_ok=True)
        immutable(output / "plan.json", plan)
        immutable_bytes(output / "source_physics_plan.json", (previous / "source_physics_plan.json").read_bytes())
        for name in ("plan.json", "summary.json", "completion.json"):
            immutable_bytes(output / f"source_7a_{name}", (previous / name).read_bytes())
        for phase in PHASES:
            run_phase(output, phase, plan["jobs"], workers)
        if not all(sha256(previous / name) == digest for name, digest in plan["previous_sha256"].items()):
            raise ValueError("Source 7a changed during generation")
        export_dataset(output, plan)
        return check(output)


def status(output):
    output = Path(output).resolve()
    plan = read_json(output / "plan.json")
    return {"step": "7b", "complete": (output / "completion.json").is_file(), "groups": len(plan["jobs"]),
            "committed_groups": {phase: sum(read_committed(output / phase / group_name(job)) is not None for job in plan["jobs"]) for phase in PHASES}}


def check(output, replay_groups=0):
    output = Path(output).resolve()
    if type(replay_groups) is not int or replay_groups < 0:
        raise ValueError("Replay group count must be a nonnegative integer")
    manifest = read_json(output / "completion.json")["sha256"]
    required = {"plan.json", "source_physics_plan.json", "summary.json", "records.jsonl", "feature_schema.json", "coverage.csv", "cases.csv", "layouts.csv"}
    required |= {f"{phase}_frozen.json" for phase in PHASES}
    required |= {f"graphs/{name}.jsonl" for name in ("train", "validation", "test", "test_unseen_k")}
    if not required.issubset(manifest):
        raise ValueError("Missing 7b export files")
    for name, digest in manifest.items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Changed/missing 7b result: {name}")
    plan = read_json(output / "plan.json")
    c = validate_settings(plan["config"])
    if plan["format"] != FORMAT or plan["jobs"] != make_jobs(c) or plan["streams"] != STREAMS:
        raise ValueError("Unexpected 7b experiment plan")
    source = read_json(output / "source_physics_plan.json")
    if sha256(output / "source_physics_plan.json") != plan["previous_sha256"]["source_physics_plan.json"]:
        raise ValueError("Copied physical source differs")
    for name in ("plan.json", "summary.json", "completion.json"):
        if sha256(output / f"source_7a_{name}") != plan["previous_sha256"][name]:
            raise ValueError("Copied 7a provenance differs")
    schema = read_json(output / "feature_schema.json")
    if schema["version"] != SCHEMA_VERSION or schema["node_features"] != NODE_FEATURES or schema["global_features"] != GLOBAL_FEATURES or schema["edge_features"] != EDGE_FEATURES:
        raise ValueError("Unexpected graph feature meanings")
    records = read_jsonl(output / "records.jsonl")
    expected_marks = {(j["layout_id"], j["n_public"], p, xi, j["split"], j["regime"])
                      for j in plan["jobs"] for p in c["power_dbm"] for xi in c["xi"]}
    marks = [(r["layout_id"], r["n_public"], r["power_dbm"], r["xi"], r["split"], r["regime"]) for r in records]
    if len(marks) != len(expected_marks) or set(marks) != expected_marks:
        raise ValueError("Case ledger duplicated, omitted or moved a scenario")
    expected_records = []
    actual_pso_evaluations = 0
    for phase in PHASES:
        frozen = read_json(output / f"{phase}_frozen.json")
        names = {f"{phase}/{group_name(job)}" for job in plan["jobs"]}
        if frozen["phase"] != phase or set(frozen["sha256"]) != names or any(manifest.get(n) != h for n, h in frozen["sha256"].items()):
            raise ValueError("Frozen group set/hash differs")
    for ji, job in enumerate(plan["jobs"]):
        search, selected, heldout = [read_committed(output / phase / group_name(job)) for phase in PHASES]
        if any(g is None or any(g[name] != value for name, value in job.items()) for g in (search, selected, heldout)):
            raise ValueError("Missing/changed committed group identity")
        if selected["search_frozen_sha256"] != sha256(output / "search_frozen.json") or heldout["selection_frozen_sha256"] != sha256(output / "selection_frozen.json"):
            raise ValueError("Search-selection-heldout chain differs")
        if len(selected["powers"]) != len(c["power_dbm"]) or len(heldout["rows"]) != len(c["power_dbm"])*len(c["xi"]):
            raise ValueError("Missing cases within a group")
        if (search["geometry_valid"] and len(search["powers"]) != len(c["power_dbm"])) or (not search["geometry_valid"] and search["powers"]):
            raise ValueError("Search power grid differs")
        evaluations_in_group = 0
        for pi, item in enumerate(selected["powers"]):
            if item["power_dbm"] != c["power_dbm"][pi]:
                raise ValueError("Candidate pool mixed total powers")
            initial = search["powers"][pi]["candidates"] if search["geometry_valid"] else []
            ids = {f"p{pi}_d{ti}_s{seed}" for ti in range(len(c["xi"])) for seed in c["pso_seeds"]} if search["geometry_valid"] and search["layout_eligible"] else set()
            if (len(initial) != len(ids) or {p["candidate_id"] for p in initial} != ids
                    or len(item["candidates"]) != len(initial)):
                raise ValueError("Missing or duplicated fixed-pool candidate")
            for before, after in zip(initial, item["candidates"]):
                fields = ("candidate_id", "origin_xi", "seed", "allocation", "search_rate", "search_feasible_by_xi")
                if any(before[name] != after[name] for name in fields) or before["evaluations"] != c["pso_evaluations"]:
                    raise ValueError("Selection changed the frozen search proposal")
                if len(before["history"]) != c["pso_evaluations"]//30 or before["history"][-1]["evaluations"] != c["pso_evaluations"]:
                    raise ValueError("PSO trace/evaluation budget differs")
                evaluations_in_group += before["evaluations"]
            if len(item["choices"]) != len(c["xi"]):
                raise ValueError("DEP choices differ from grid")
            for ti, choice in enumerate(item["choices"]):
                if choice["xi"] != c["xi"][ti]:
                    raise ValueError("Changed DEP order")
                winner = choose_label(item["candidates"], ti, choice["xi"])
                if choice["selected_candidate_id"] != (None if winner is None else winner["candidate_id"]) or choice["allocation"] != (None if winner is None else winner["allocation"]):
                    raise ValueError("Label differs from declared fixed-pool rule")
                row = heldout["rows"][pi * len(c["xi"]) + ti]
                if any(row[name] != value for name, value in choice.items()) or row["heldout_used_for_label_filter"] is not False:
                    raise ValueError("Heldout filtered/changed a selected label")
                graph = search["powers"][pi]["graphs"][ti] if search["geometry_valid"] else None
                if row["graph"] != graph or row["power_dbm"] != item["power_dbm"] or any(row[name] != value for name, value in job.items()):
                    raise ValueError("Graph or case identity differs from search input")
                if row["label_available"] != (row["allocation"] is not None) or (row["heldout"] is not None) != row["label_available"]:
                    raise ValueError("Label/audit availability differs")
                if row["graph"] is not None and validate_graph(row["graph"]) != job["n_public"]:
                    raise ValueError("Graph K differs")
                if row["regime"] == "unseen_k" and row["split"] != "test":
                    raise ValueError("Unseen K crossed into train/validation")
                if winner is not None:
                    actual = at_target(winner["selection"], choice["xi"])
                    if not actual["passed"]:
                        raise ValueError("Selected label failed its independent selection")
                    if row["selection"] != winner["selection"] or row["search_rate"] != winner["search_rate"]:
                        raise ValueError("Selected label score differs from frozen winner")
                if row["heldout"] is not None:
                    actual = at_target(row["heldout"], row["xi"])
                    if any(row["heldout"][name] != value for name, value in actual.items()):
                        raise ValueError("Heldout accepted flag differs from saved checks")
        if search["pso_evaluations"] != evaluations_in_group:
            raise ValueError("Group PSO evaluation total differs from traces")
        actual_pso_evaluations += evaluations_in_group
        expected_records.extend(heldout["rows"])
        if ji < replay_groups:
            replay_group(plan, source, job, search, selected, heldout)
    if expected_records != records or len(records) != plan["planned_cases"]:
        raise ValueError("Full case ledger differs from heldout records")
    for name in ("train", "validation", "test", "test_unseen_k"):
        exported = read_jsonl(output / "graphs" / f"{name}.jsonl")
        matching = [r for r in records if role(r) == name and r["label_available"]]
        if len(exported) != len(matching):
            raise ValueError("Graph export removed or added labels")
        for actual, row in zip(exported, matching):
            if any(value != row.get(key) for key, value in actual.items() if key != "heldout_passed") or actual["heldout_passed"] != row["heldout"]["passed"]:
                raise ValueError("Graph export content differs from original label")
    counts, coverage, ready = summaries(plan, records)
    result = read_json(output / "summary.json")
    if (result["counts"] != counts or result["training_ready_minimum_only"] != ready or result["heldout_filtered_labels"]
            or result["normalization_fitted"] or result["gnn_trained"] or not result["complete"]):
        raise ValueError("Summary differs from case ledger")
    if result["pso_evaluations"] != actual_pso_evaluations or result["planned_cases"] != len(expected_marks):
        raise ValueError("Summary case or PSO budget differs")
    return result | {"verified_files": len(manifest), "replayed_groups": min(replay_groups, len(plan["jobs"]))}


def replay_group(plan, source, job, search, selected, heldout):
    """Recompute saved selection/heldout statistics, without changing output."""
    def compare(actual, expected):
        if isinstance(actual, dict):
            for key in actual:
                compare(actual[key], expected[key])
        elif isinstance(actual, list):
            if len(actual) != len(expected):
                raise ValueError("Replay list size differs")
            for a, b in zip(actual, expected):
                compare(a, b)
        elif isinstance(actual, (int, float)) and not isinstance(actual, bool):
            np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-12)
        elif actual != expected:
            raise ValueError("Replayed physical decision differs")
    c = plan["config"]
    if search["geometry_valid"] and search["layout_eligible"]:
        physical, layout, bank = context(plan, source, job, "selection")
        for item in selected["powers"]:
            for proposal in item["candidates"]:
                actual = raw_assessment(dict(physical, total_dbm=item["power_dbm"]), layout, bank,
                                        np.asarray(selected["order"], int), proposal["allocation"],
                                        len(c["power_dbm"])*len(c["xi"])*len(c["pso_seeds"]), c["alpha"]/(2*len(plan["jobs"])))
                compare(actual, proposal["selection"])
        del bank
    if any(row["label_available"] for row in heldout["rows"]):
        physical, layout, bank = context(plan, source, job, "heldout")
        for row in heldout["rows"]:
            if row["label_available"]:
                actual = raw_assessment(dict(physical, total_dbm=row["power_dbm"]), layout, bank,
                                        np.asarray(selected["order"], int), row["allocation"],
                                        len(c["power_dbm"])*len(c["xi"]), c["alpha"]/(2*len(plan["jobs"])))
                compare(actual, row["heldout"])
        del bank
    print(f"7b replay matches: layout={job['layout_id']} K={job['n_public']}", flush=True)
