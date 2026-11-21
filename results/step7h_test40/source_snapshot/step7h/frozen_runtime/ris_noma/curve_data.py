"""7g nested geometry corpus. Reuses frozen 7e physics/PSO without changing it."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
import json
import multiprocessing as mp
import platform

from filelock import FileLock
import numpy as np
import scipy
import torch

from .continuous_data import STREAMS as LABEL_STREAMS, parameter_cases, geometry_from
from .continuous_dataset import freeze, read_lines
from .continuous_learning import check as check_7f
from .continuous_workers import search_job, select_job, audit_job, read_group, choose_label
from .expanded_data import write_table
from .learning_data import read_json, sha256
from .multik_benchmark import verify_manifest
from .multik_core import physical_allocation
from .multik_data_core import at_target
from .multik_dataset import immutable, immutable_bytes, write_json, receipt_path, read_committed
from .multik_graph import fit_normalizer, validate_graph
from .multik_learning import validate_config as validate_fit_config
from .validation import family_constants

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "nested_learning_corpus_7g_v1"
NUMERIC_FILES = ("run_step7g.py", "ris_noma/curve_data.py", "ris_noma/curve_models.py",
                 "ris_noma/curve_fit.py", "ris_noma/curve_physics.py",
                 "ris_noma/curve_learning.py", "ris_noma/curve_analysis.py")
COPIES = {"source_7e_plan.json": ("dataset", "plan.json"),
          "source_7e_completion.json": ("dataset", "completion.json"),
          "source_physics_plan.json": ("dataset", "source_physics_plan.json"),
          "source_dataset_plan.json": ("dataset", "source_dataset_plan.json"),
          "source_7f_plan.json": ("training", "plan.json"),
          "source_7f_selection.json": ("training", "selection_frozen.json"),
          "source_7f_summary.json": ("training", "summary.json"),
          "source_7f_completion.json": ("training", "completion.json")}
FIELDS = {"scope", "training_layout_ids", "training_sizes", "validation_layout_ids",
          "reserved_test_layout_ids", "training_public_counts", "reserved_unseen_public_counts",
          "cases_per_layout", "parameter_seed", "power_dbm_range", "dep_range", "label",
          "training", "architectures", "physics", "curve"}


def ordered_ints(value, minimum=0):
    return (isinstance(value, list) and bool(value)
            and all(type(v) is int and v >= minimum for v in value)
            and value == sorted(set(value)))


def validate_settings(settings):
    c = deepcopy(settings)
    if set(c) != FIELDS or c["scope"] not in ("smoke", "development_pilot"):
        raise ValueError("Unexpected 7g settings/scope")
    for name in ("training_layout_ids", "validation_layout_ids", "reserved_test_layout_ids"):
        if not ordered_ints(c[name], 1000):
            raise ValueError("Use nonempty sorted distinct layout IDs >= 1000")
    ids = sum((c[n] for n in ("training_layout_ids", "validation_layout_ids", "reserved_test_layout_ids")), [])
    if len(ids) != len(set(ids)):
        raise ValueError("Train, validation and reserved test layouts must be disjoint")
    if (not ordered_ints(c["training_sizes"], 1)
            or c["training_sizes"][-1] != len(c["training_layout_ids"])):
        raise ValueError("Nested sizes must increase and end at the declared train-layout count")
    for name in ("training_public_counts", "reserved_unseen_public_counts"):
        if not ordered_ints(c[name], 1):
            raise ValueError("Public counts must be sorted distinct positive integers")
    if set(c["training_public_counts"]) & set(c["reserved_unseen_public_counts"]):
        raise ValueError("Reserved unseen K cannot enter train/validation")
    for name in ("power_dbm_range", "dep_range"):
        v = c[name]
        if (not isinstance(v, list) or len(v) != 2
                or any(type(x) not in (int, float) or not np.isfinite(x) for x in v) or v[0] >= v[1]):
            raise ValueError("Require increasing finite power/DEP ranges")
    if not 0 < c["dep_range"][0] < c["dep_range"][1] < 1:
        raise ValueError("DEP must lie in (0,1)")
    if type(c["cases_per_layout"]) is not int or not 1 <= c["cases_per_layout"] <= 1000:
        raise ValueError("Invalid cases_per_layout")
    if type(c["parameter_seed"]) is not int or c["parameter_seed"] < 0:
        raise ValueError("Invalid parameter seed")
    label = c["label"]
    if set(label) != {"search_samples", "selection_samples", "audit_samples", "pso_seeds",
                      "pso_evaluations", "search_dep_margin", "search_qos_margin", "alpha"}:
        raise ValueError("Unexpected label settings")
    for name in ("search_samples", "selection_samples", "audit_samples"):
        if type(label[name]) is not int or label[name] < 1000:
            raise ValueError("Label banks require >=1000 samples")
    if not ordered_ints(label["pso_seeds"]):
        raise ValueError("Use distinct ordered PSO seeds")
    if (type(label["pso_evaluations"]) is not int or label["pso_evaluations"] < 30
            or label["pso_evaluations"] % 30):
        raise ValueError("PSO budget includes initialization and must be a multiple of 30")
    for name in ("search_dep_margin", "search_qos_margin"):
        if type(label[name]) not in (int, float) or not np.isfinite(label[name]) or label[name] < 0:
            raise ValueError("Margins must be finite and nonnegative")
    if c["dep_range"][1] + label["search_dep_margin"] >= 1:
        raise ValueError("DEP target plus margin must be below one")
    for alpha in (label["alpha"], c["physics"].get("alpha")):
        if type(alpha) not in (int, float) or not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("Require 0 < alpha < 1")
    if set(c["architectures"]) != {"reference", "small"}:
        raise ValueError("Declare the reference and small architectures")
    for architecture in c["architectures"].values():
        if set(architecture) != {"hidden", "layers", "mlp_inner"}:
            raise ValueError("Unexpected architecture fields")
        validate_fit_config(dict(c["training"], **architecture, validation_physics_samples=1000,
                                 test_physics_samples=1000, alpha=c["physics"]["alpha"]))
    p = c["physics"]
    if set(p) != {"validation_samples", "train_probe_samples", "train_probe_layout_count", "alpha"}:
        raise ValueError("Unexpected physical-evaluation settings")
    for name in ("validation_samples", "train_probe_samples"):
        if type(p[name]) is not int or p[name] < 1000:
            raise ValueError("Physical banks require >=1000 samples")
    if (type(p["train_probe_layout_count"]) is not int
            or not 1 <= p["train_probe_layout_count"] <= c["training_sizes"][0]):
        raise ValueError("Train probe must be a fixed subset of every train prefix")
    curve = c["curve"]
    if set(curve) != {"bootstrap_resamples", "bootstrap_seed", "minimum_rate_relative_gain",
                     "minimum_pass_fraction_gain", "minimum_mse_relative_gain"}:
        raise ValueError("Unexpected learning-curve settings")
    if type(curve["bootstrap_resamples"]) is not int or curve["bootstrap_resamples"] < 100:
        raise ValueError("Require >=100 layout bootstrap resamples")
    if type(curve["bootstrap_seed"]) is not int or curve["bootstrap_seed"] < 0:
        raise ValueError("Invalid bootstrap seed")
    for name in ("minimum_rate_relative_gain", "minimum_pass_fraction_gain", "minimum_mse_relative_gain"):
        if type(curve[name]) not in (int, float) or not np.isfinite(curve[name]) or not 0 <= curve[name] < 1:
            raise ValueError("Practical gain thresholds must lie in [0,1)")
    return c


def fit_config(c, size):
    return dict(c["training"], **c["architectures"][size],
                validation_physics_samples=c["physics"]["validation_samples"],
                test_physics_samples=c["physics"]["train_probe_samples"], alpha=c["physics"]["alpha"])


def geometry_spec(c, source_plan):
    return {"config": dict(c["label"], cases_per_layout=c["cases_per_layout"],
                           parameter_seed=c["parameter_seed"], power_dbm_range=c["power_dbm_range"],
                           dep_range=c["dep_range"], profile=source_plan["config"]["profile"]),
            "deployment": source_plan["deployment"], "array_settings": source_plan["array_settings"]}


def jobs_for(c, root_seed):
    # Reserved tests are deliberately absent from this function.
    return [{"id": f"{split}_l{lid:06d}_k{k}", "layout_id": lid, "n_public": k,
             "split": split, "regime": "seen_k", "cases": parameter_cases(c, root_seed, lid)}
            for split, ids in (("train", c["training_layout_ids"]), ("validation", c["validation_layout_ids"]))
            for lid in ids for k in c["training_public_counts"]]


def stage_jobs(plan, size):
    if size not in plan["config"]["training_sizes"]:
        raise ValueError("Stage must be one of the predeclared nested train sizes")
    ids = set(plan["config"]["training_layout_ids"][:size])
    return [j for j in plan["jobs"] if j["split"] == "validation" or j["layout_id"] in ids]


def source_hashes(dataset, training):
    roots = {"dataset": Path(dataset), "training": Path(training)}
    return {copy: sha256(roots[role]/name) for copy, (role, name) in COPIES.items()}


def inspect(dataset, training, settings, workers=2):
    dataset, training = Path(dataset).resolve(), Path(training).resolve()
    baseline = check_7f(training, dataset)
    c = validate_settings(settings)
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use 1..4 workers; 2 recommended")
    old = read_json(dataset/"plan.json")
    old_fit = read_json(training/"plan.json")
    source = read_json(dataset/"source_physics_plan.json")
    geometry = read_json(dataset/"source_dataset_plan.json")
    source_7d = read_json(dataset/"source_7d_plan.json")
    if baseline["scope"] != "development_pilot":
        raise ValueError("7g requires the main 7f result, not smoke")
    previous_ids = {x for values in old["config"]["splits"].values() for x in values}
    previous_ids.update(x for values in geometry["config"]["splits"].values() for x in values)
    previous_ids.update(source_7d["config"]["calibration_layout_ids"])
    previous_ids.update(source_7d["config"]["benchmark_layout_ids"])
    previous_ids.update(j["layout_id"] for j in source.get("layouts", []) if isinstance(j, dict) and "layout_id" in j)
    if c["training_layout_ids"] and set(sum((c[n] for n in
            ("training_layout_ids", "validation_layout_ids", "reserved_test_layout_ids")), [])) & previous_ids:
        raise ValueError("7g IDs overlap previous train/validation/test/development layouts")
    if c["power_dbm_range"] != old["config"]["power_dbm_range"] or c["dep_range"] != old["config"]["dep_range"]:
        raise ValueError("7g compares data/model size in the unchanged continuous physical domain")
    if c["architectures"]["reference"] != {k: old_fit["config"][k] for k in ("hidden", "layers", "mlp_inner")}:
        raise ValueError("Reference architecture must match 7f")
    if c["training"]["loss_weights"] != old_fit["config"]["loss_weights"]:
        raise ValueError("Keep the 7f weighted loss while testing data/model size")
    if c["scope"]!="smoke" and (c["training_public_counts"]!=old_fit["training_public_counts"]
            or c["reserved_unseen_public_counts"]!=old_fit["unseen_public_counts"]):
        raise ValueError("Keep the 7f train/unseen K split in the main data-size comparison")
    if c["scope"] != "smoke" and any(c["label"][k] != old["config"][k] for k in c["label"] if k != "alpha"):
        raise ValueError("Main 7g label physics/budgets/margins must match 7e")
    root_seed = old["root_seed"]
    jobs = jobs_for(c, root_seed)
    spec = geometry_spec(c, old)
    # Inspect train/validation geometry only; never build reserved-test geometry.
    for job in jobs:
        geometry_from(spec, source, geometry, job, c["label"]["search_samples"])
    from .multik_learning_models import build_model
    parameters = {size: {kind: sum(p.numel() for p in build_model(
        kind, source["base_config"]["bounds"], fit_config(c, size)).parameters())
        for kind in ("gnn", "mlp")} for size in ("reference", "small")}
    if any(abs(v["gnn"]-v["mlp"])/v["gnn"] > .01 for v in parameters.values()):
        raise ValueError("GNN/MLP parameter counts must match within 1% at each size")
    if any(parameters["small"][k] >= parameters["reference"][k] for k in ("gnn", "mlp")):
        raise ValueError("Small models must have fewer parameters")
    files = tuple(dict.fromkeys((*old_fit["code_sha256"], *NUMERIC_FILES)))
    count = len(c["training_sizes"])*2*2*len(c["training"]["seeds"])
    return {"step": "7g", "format": FORMAT, "config": c, "workers": workers,
            "root_seed": root_seed, "jobs": jobs, "geometry_spec": spec,
            "label_streams": LABEL_STREAMS, "pso_settings": old["pso_settings"],
            "source_sha256": source_hashes(dataset, training),
            "code_sha256": {name: sha256(ROOT/name) for name in files},
            "versions": {"system": platform.system(), "python": platform.python_version(),
                "numpy": np.__version__, "scipy": scipy.__version__, "torch": str(torch.__version__),
                "filelock": version("filelock")},
            "model_parameters": parameters, "max_fits": count,
            "max_validation_candidates": count*(len(c["training"]["snapshot_epochs"])+1),
            "max_train_probe_views": count*2, "reserved_test_generated": False,
            "test_used_for_selection": False, "physical_model_changed": False,
            "label_audit_used_for_filtering": False,
            "normalization_policy": "fit separately on labeled train graphs of each prefix; shared by all models in that prefix",
            "validation_policy": "same raw validation geometries/cases/banks at every train size; candidate family covers all predeclared sizes",
            "final_test_policy": "reserved layout IDs only; no test graphs, labels, channels or predictions in 7g"}


def initialize(corpus, dataset, training, settings, workers=2, resume=False):
    corpus = Path(corpus).resolve()
    for source in (Path(dataset).resolve(), Path(training).resolve()):
        if source == corpus or source in corpus.parents or corpus in source.parents:
            raise ValueError("Corpus must be separate from source data/results")
    plan = inspect(dataset, training, settings, workers)
    if corpus.exists() and any(corpus.iterdir()):
        if not resume:
            raise ValueError("Corpus is nonempty; use --resume with the same plan")
        if read_json(corpus/"plan.json") != plan:
            raise ValueError("Changed 7g plan/code/source/runtime/workers; use a new corpus")
    corpus.mkdir(parents=True, exist_ok=True)
    immutable(corpus/"plan.json", plan)
    roots = {"dataset": Path(dataset), "training": Path(training)}
    for copy, (role, name) in COPIES.items():
        immutable_bytes(corpus/copy, (roots[role]/name).read_bytes())
    for name in plan["code_sha256"]:
        immutable_bytes(corpus/"source_snapshot"/name, (ROOT/name).read_bytes())
    return plan


def label_folder(corpus, job):
    return Path(corpus)/"labels"/job["id"]


def label_plan(plan, job):
    # A job-local immutable barrier permits appending new layouts without
    # invalidating earlier selections. Alpha is divided by the MAX job count,
    # so adding prefixes does not change the certification of existing labels.
    config = dict(plan["geometry_spec"]["config"], alpha=plan["config"]["label"]["alpha"]/len(plan["jobs"]))
    return {"step": "7g_label", "jobs": [job], "config": config,
            "deployment": plan["geometry_spec"]["deployment"],
            "array_settings": plan["geometry_spec"]["array_settings"],
            "pso_settings": plan["pso_settings"], "corpus_plan_sha256": plan["plan_sha256"]}


def label_task(arguments):
    corpus, job = arguments
    corpus = Path(corpus)
    plan = dict(read_json(corpus/"plan.json"), plan_sha256=sha256(corpus/"plan.json"))
    folder = label_folder(corpus, job)
    local = label_plan(plan, job)
    folder.mkdir(parents=True, exist_ok=True)
    immutable(folder/"plan.json", local)
    for name in ("source_physics_plan.json", "source_dataset_plan.json"):
        immutable_bytes(folder/name, (corpus/name).read_bytes())
    if (folder/"completion.json").exists():
        verify_manifest(folder, read_json(folder/"completion.json")["sha256"])
        return job["id"]
    for phase, function in (("search", search_job), ("selection", select_job), ("audit", audit_job)):
        function((str(folder), job))
        freeze(folder, phase, local)
    files = [p for p in sorted(folder.rglob("*")) if p.is_file() and p.name != "completion.json" and not p.name.endswith(".tmp")]
    immutable(folder/"completion.json", {"sha256": {p.relative_to(folder).as_posix(): sha256(p) for p in files}})
    return job["id"]


def execute_jobs(function, arguments, workers, label):
    arguments = list(arguments)
    print(f"7g {label}: {len(arguments)} groups pending", flush=True)
    if workers == 1:
        for i, arg in enumerate(arguments, 1):
            name = function(arg)
            print(f"7g {label} {i}/{len(arguments)}: {name}", flush=True)
    elif arguments:
        todo = iter(arguments)
        active = {}
        done = 0
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
        def submit():
            arg = next(todo, None)
            if arg is not None:
                active[executor.submit(function, arg)] = arg
        try:
            for _ in range(workers):
                submit()
            while active:
                finished, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in finished:
                    name = future.result()
                    del active[future]
                    done += 1
                    print(f"7g {label} {done}/{len(arguments)}: {name}", flush=True)
                    submit()
        finally:
            for future in active:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)


def checked_label_rows(corpus, plan, job):
    folder = label_folder(corpus, job)
    verify_manifest(folder, read_json(folder/"completion.json")["sha256"])
    for name in ("source_physics_plan.json","source_dataset_plan.json"):
        if sha256(folder/name)!=sha256(Path(corpus)/name):
            raise ValueError("Job-local physics/geometry source changed")
    if read_json(folder/"plan.json") != label_plan(dict(plan, plan_sha256=sha256(Path(corpus)/"plan.json")), job):
        raise ValueError("Job-local plan/alpha differs from corpus")
    search, selection, audit = [read_group(folder, p, job) for p in ("search", "selection", "audit")]
    if any(doc is None or doc["job"] != job for doc in (search, selection, audit)):
        raise ValueError("Missing/wrong label phase")
    if selection["search_frozen_sha256"] != sha256(folder/"search_frozen.json") or audit["selection_frozen_sha256"] != sha256(folder/"selection_frozen.json"):
        raise ValueError("Label barrier chain differs")
    if any(len(doc[key]) != len(job["cases"]) for doc, key in ((search, "cases"), (selection, "rows"), (audit, "rows"))):
        raise ValueError("Missing planned label case")
    for phase, doc in zip(("search", "selection", "audit"), (search, selection, audit)):
        meta = doc["metadata"]
        if (meta["bank_entropy"] != [plan["root_seed"], job["layout_id"], job["n_public"], LABEL_STREAMS[phase]]
                or meta["bank_samples"] != plan["config"]["label"][phase+"_samples"]
                or meta["receiver_xy"] != search["metadata"]["receiver_xy"]
                or meta["los_sha256"] != search["metadata"]["los_sha256"]):
            raise ValueError("Label bank/geometry changed")
    alpha = plan["config"]["label"]["alpha"]/(2*len(plan["jobs"]))
    family = len(job["cases"])*len(plan["config"]["label"]["pso_seeds"])*5
    rows = []
    physics = read_json(Path(corpus)/"source_physics_plan.json")
    for expected, case, chosen, final in zip(job["cases"], search["cases"], selection["rows"], audit["rows"]):
        if any(doc[k] != v for k, v in expected.items() for doc in (case, chosen, final)):
            raise ValueError("Continuous case changed between label stages")
        identity = job["id"]+f"_c{expected['case_index']:03d}"
        if any(doc["case_id"] != identity for doc in (case, chosen, final)):
            raise ValueError("Case identity changed")
        if validate_graph(case["graph"]) != job["n_public"]:
            raise ValueError("Wrong graph K")
        if np.argsort(np.asarray(case["graph"]["node_raw"])[1:,9]).tolist()!=search["order"]:
            raise ValueError("Graph SIC order differs from label evaluator")
        candidates = [x for pool in case["pools"] for x in pool["candidates"]]
        settings=plan["config"]["label"]
        if [p["seed"] for p in case["pools"]] != (settings["pso_seeds"] if search["layout_eligible"] else []):
            raise ValueError("Missing PSO seed run")
        for pool in case["pools"]:
            if (pool["objective_evaluations"]!=settings["pso_evaluations"]
                    or len(pool["history"])!=settings["pso_evaluations"]//30
                    or len(pool["candidates"])>5):
                raise ValueError("Wrong PSO evaluation/archive budget")
        if len(candidates) != len(chosen["candidates"]):
            raise ValueError("Selection candidates missing")
        for original, scored in zip(candidates, chosen["candidates"]):
            if any(scored[k] != v for k, v in original.items()):
                raise ValueError("Selection changed a PSO proposal")
            if scored["physical"]["constants"] != family_constants(plan["config"]["label"]["selection_samples"], family, job["n_public"], alpha):
                raise ValueError("Wrong label confidence family")
            if any(scored[k] != v for k, v in at_target(scored["physical"], case["xi"]).items()):
                raise ValueError("Label acceptance differs")
        winner = choose_label(chosen["candidates"])
        if chosen["selected_candidate_id"] != (None if winner is None else winner["candidate_id"]):
            raise ValueError("Wrong label-selection rule")
        if chosen["label_available"] != (winner is not None) or chosen["allocation"] != (None if winner is None else winner["allocation"]):
            raise ValueError("Label value/availability differs")
        if any(final[k] != chosen[k] for k in ("allocation", "label_available", "selected_candidate_id")) or final["audit_used_for_label_filter"] is not False:
            raise ValueError("Audit filtered/changed a label")
        if chosen["label_available"]:
            physical_allocation(chosen["allocation"], job["n_public"], physics["base_config"]["bounds"])
        rows.append({"case_id": identity, "layout_id": job["layout_id"], "n_public": job["n_public"],
                     "split": job["split"], "regime": "seen_k", "role": job["split"],
                     "power_dbm": case["power_dbm"], "xi": case["xi"], "graph": case["graph"],
                     "order": search["order"], "allocation": chosen["allocation"],
                     "has_label": chosen["label_available"]})
    return rows


def stage_folder(corpus, size):
    return Path(corpus)/"stages"/f"train_{size:04d}"


def label_coverage(corpus,plan,size,rows):
    counts={}
    groups=[]
    for role in ("train","validation"):
        chosen_rows=[r for r in rows if r["split"]==role]
        reasons=Counter()
        audits=0
        for job in [j for j in stage_jobs(plan,size) if j["split"]==role]:
            folder=label_folder(corpus,job)
            selected=read_group(folder,"selection",job)
            audit=read_group(folder,"audit",job)
            search=read_group(folder,"search",job)
            reasons.update(r["missing_reason"] for r in selected["rows"] if not r["label_available"])
            passed=sum(r["audit"] is not None and r["audit"]["passed"] for r in audit["rows"])
            audits+=passed
            groups.append({"split":role,"layout_id":job["layout_id"],"n_public":job["n_public"],
                "planned":len(job["cases"]),"labels":sum(r["label_available"] for r in selected["rows"]),
                "teacher_audit_passed":passed,"search_layout_gain_rule_passed":search["layout_eligible"],
                "objective_evaluations":search["objective_evaluations"]})
        counts[role]={"planned_layouts":len({r["layout_id"] for r in chosen_rows}),
            "labeled_layouts":len({r["layout_id"] for r in chosen_rows if r["has_label"]}),
            "missing_label_reasons":dict(reasons),"teacher_audit_passed":audits}
    return counts,groups


def export_stage(corpus, plan, size):
    folder = stage_folder(corpus, size)
    rows = [row for job in stage_jobs(plan, size) for row in checked_label_rows(corpus, plan, job)]
    train = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "validation"]
    labeled = [r for r in train if r["has_label"]]
    if not labeled or not any(r["has_label"] for r in validation):
        raise ValueError("Stage needs labeled train and validation cases")
    for name, values in (("train", train), ("validation", validation)):
        immutable_bytes(folder/(name+".jsonl"), "".join(json.dumps(r, ensure_ascii=False, allow_nan=False)+"\n" for r in values).encode("utf-8"))
    immutable(folder/"normalization.json", {"fit_role": "train", "training_graph_count": len(labeled),
                                           "parameters": fit_normalizer([r["graph"] for r in labeled])})
    groups = {job["id"]: sha256(label_folder(corpus, job)/"completion.json") for job in stage_jobs(plan, size)}
    coverage,group_rows=label_coverage(corpus,plan,size,rows)
    write_table(folder/"label_coverage_by_group.csv",group_rows)
    info = {"step": "7g", "train_layouts": size, "validation_layouts": len(plan["config"]["validation_layout_ids"]),
            "corpus_plan_sha256": sha256(Path(corpus)/"plan.json"),
            "planned_counts": {"train": len(train), "validation": len(validation)},
            "labeled_counts": {"train": len(labeled), "validation": sum(r["has_label"] for r in validation)},
            "coverage":coverage,
            "label_group_completions": groups, "audit_used_to_filter_labels": False,
            "reserved_test_generated": False}
    immutable(folder/"summary.json", info)
    immutable(folder/"completion.json", {"sha256": {p.name: sha256(p) for p in folder.iterdir() if p.is_file() and p.name != "completion.json"}})
    return info


def check_corpus(corpus, size=None):
    corpus = Path(corpus).resolve()
    plan = read_json(corpus/"plan.json")
    validate_settings(plan["config"])
    if plan["format"] != FORMAT or plan["reserved_test_generated"] or plan["test_used_for_selection"]:
        raise ValueError("Wrong corpus format/test policy")
    verify_manifest(ROOT, plan["code_sha256"])
    verify_manifest(corpus/"source_snapshot", plan["code_sha256"])
    for name, digest in plan["source_sha256"].items():
        if sha256(corpus/name) != digest:
            raise ValueError("Copied source provenance changed")
    if jobs_for(plan["config"], plan["root_seed"]) != plan["jobs"]:
        raise ValueError("Nested jobs/continuous schedule changed")
    stages = [size] if size is not None else [n for n in plan["config"]["training_sizes"] if (stage_folder(corpus,n)/"completion.json").exists()]
    checked = []
    baseline_validation = None
    for n in stages:
        folder = stage_folder(corpus, n)
        verify_manifest(folder, read_json(folder/"completion.json")["sha256"])
        expected_rows = [r for j in stage_jobs(plan,n) for r in checked_label_rows(corpus,plan,j)]
        train = read_lines(folder/"train.jsonl")
        validation = read_lines(folder/"validation.jsonl")
        if train != [r for r in expected_rows if r["split"]=="train"] or validation != [r for r in expected_rows if r["split"]=="validation"]:
            raise ValueError("Stage differs from committed label cases")
        if baseline_validation is None:
            baseline_validation = validation
        elif validation != baseline_validation:
            raise ValueError("Raw validation changed between nested prefixes")
        labeled = [r for r in train if r["has_label"]]
        if read_json(folder/"normalization.json") != {"fit_role":"train","training_graph_count":len(labeled),
                "parameters":fit_normalizer([r["graph"] for r in labeled])}:
            raise ValueError("Prefix normalizer is not train-only")
        info=read_json(folder/"summary.json")
        if info["corpus_plan_sha256"] != sha256(corpus/"plan.json") or info["train_layouts"] != n:
            raise ValueError("Stage lineage changed")
        if info["coverage"]!=label_coverage(corpus,plan,n,expected_rows)[0]:
            raise ValueError("Missing-label coverage/geometry assumptions changed")
        checked.append(info)
    return {"step":"7g","corpus_valid":True,"stages":checked,"reserved_test_generated":False}


def prepare(corpus, dataset, training, settings, size, workers=2, resume=False):
    corpus=Path(corpus).resolve()
    corpus.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(corpus)+".lock",timeout=0):
        plan=initialize(corpus,dataset,training,settings,workers,resume)
        jobs=stage_jobs(plan,size)
        pending=[j for j in jobs if not (label_folder(corpus,j)/"completion.json").exists()]
        execute_jobs(label_task,[(str(corpus),j) for j in pending],workers,f"labels train={size}")
        export_stage(corpus,plan,size)
        return check_corpus(corpus,size)
