"""Bind completed Windows 7g artifacts without following their old absolute paths."""
from pathlib import Path
from collections import defaultdict
from shutil import copy2
import numpy as np

from .common import read, lines, digest, verify, safe_path, write, object_hash
from ris_noma.curve_models import load_model
from ris_noma.multik_graph import fit_normalizer, validate_graph


def bind(corpus, training, diagnostics, thorough=True):
    corpus, training, diagnostics = (Path(p).resolve() for p in (corpus, training, diagnostics))
    stage = training / "stages/train_0032"
    cp, sp = read(corpus / "plan.json"), read(stage / "plan.json")
    dp, ds, dm = (read(diagnostics / name) for name in ("plan.json", "selection.json", "models.json"))
    if cp.get("format") != "nested_learning_corpus_7g_v1" or sp.get("train_layouts") != 32:
        raise ValueError("Require the completed main 7g/32 corpus and training stage")
    if cp["config"]["training_public_counts"] != [2, 3, 4, 5] or cp["config"]["reserved_unseen_public_counts"] != [6]:
        raise ValueError("The frozen train-K / unseen-K split differs")
    for path in (stage / "completion.json", diagnostics / "completion.json"):
        if not path.is_file(): raise ValueError(f"Source is incomplete: {path}")
    if thorough:
        verify(stage, read(stage / "completion.json"))
        verify(diagnostics, read(diagnostics / "completion.json"))
        corpus_stage = corpus / "stages/train_0032/complete.json"
        if not corpus_stage.exists(): corpus_stage = corpus / "stages/train_0032/completion.json"
        if corpus_stage.exists(): verify(corpus_stage.parent, read(corpus_stage))
    if (dp.get("format") != "supplement_7g32_v1" or dp.get("scope") != "development_diagnostic"
            or dp.get("reserved_test_generated") is not False or ds.get("test_used_for_selection") is not False):
        raise ValueError("Require completed main 7g supplementary diagnostics without final test")
    if (dp["source_corpus_plan_sha256"] != digest(corpus / "plan.json")
            or dp["source_stage_completion_sha256"] != digest(stage / "completion.json")
            or dp["source_selection_sha256"] != digest(stage / "selection_frozen.json")):
        raise ValueError("7g diagnostics belong to different source data/training")
    train, validation = (lines(stage / f"inputs/{role}.jsonl") for role in ("train", "validation"))
    for role, rows in (("train", train), ("validation", validation)):
        if digest(stage / f"inputs/{role}.jsonl") != dp["source_input_sha256"][role]:
            raise ValueError("Source train/validation inputs changed")
        if any(r["split"] != role or r["n_public"] not in (2, 3, 4, 5) for r in rows):
            raise ValueError("Unexpected split or K in training inputs")
        for row in rows: validate_graph(row["graph"])
    ids = {role: sorted({r["layout_id"] for r in rows}) for role, rows in (("train", train), ("validation", validation))}
    reserved = cp["config"]["reserved_test_layout_ids"]
    if len(ids["train"]) != 32 or set(ids["train"]) & set(ids["validation"]) or (set(ids["train"]) | set(ids["validation"])) & set(reserved):
        raise ValueError("Source layouts overlap or main prefix is incomplete")
    norm = fit_normalizer([r["graph"] for r in train if r["has_label"]])
    if norm != sp["normalization"]: raise ValueError("Source normalizer is not labeled-train-only")
    original = read(stage / "selection_frozen.json")
    refs = {}
    for kind in ("gnn", "mlp"):
        for seed in (7, 17, 27):
            name = f"small::{kind}_s{seed}_physical"
            ref = original["views"][name]
            path = safe_path(stage, ref["path"])
            key = "frozen32::" + name
            diagnostic_ref = dm[key]
            if digest(path) != ref["sha256"] or diagnostic_ref["sha256"] != ref["sha256"]:
                raise ValueError("Checkpoint hashes disagree between original and diagnostic selections")
            selected = ds["frozen"][key]["corrected"]
            delta = float(selected.rsplit("::d", 1)[1])
            if delta not in dp["settings"]["deltas"]: raise ValueError("Delta was not a declared validation candidate")
            _, saved = load_model(path)
            if (saved["training_layout_ids"] != ids["train"] or saved["trained_public_counts"] != [2, 3, 4, 5]
                    or saved["normalization"] != norm or saved["model_kind"] != kind or saved["seed"] != seed
                    or saved["epoch"] != ref["epoch"] or saved["architecture_name"] != "small"):
                raise ValueError("Checkpoint metadata or training-only normalization differs")
            refs[f"{kind}_s{seed}"] = {"kind": kind, "seed": seed, "epoch": ref["epoch"], "path": str(path),
                "sha256": ref["sha256"], "delta": delta, "normalization": norm,
                "architecture": "small", "parameters": ref["parameters"], "source_view": key}
    low = {}
    for row in train:
        if row["n_public"] == 2:
            low[row["layout_id"]] = float(row["graph"]["global_raw"][18])
    low_ids = sorted(lid for lid, cv in low.items() if cv < .9)
    return {"corpus": corpus, "training": training, "diagnostics": diagnostics, "stage": stage,
            "corpus_plan": cp, "stage_plan": sp, "diagnostic_plan": dp,
            "train": train, "validation": validation, "layout_ids": ids,
            "models": refs, "low_cv_ids": low_ids, "train_cv": low,
            "physics": read(stage / "source_physics_plan.json"),
            "dataset": read(stage / "source_dataset_plan.json"),
            "source_hashes": {"corpus_plan": digest(corpus / "plan.json"), "stage_complete": digest(stage / "completion.json"),
                "diagnostic_complete": digest(diagnostics / "completion.json"), "diagnostic_selection": digest(diagnostics / "selection.json"),
                "train_inputs": digest(stage / "inputs/train.jsonl"), "validation_inputs": digest(stage / "inputs/validation.jsonl")}}


def snapshot(output, source):
    output = Path(output)
    for name, data in (("corpus_plan", source["corpus_plan"]), ("stage_plan", source["stage_plan"]),
                       ("physics", source["physics"]), ("dataset", source["dataset"])):
        write(output / f"inputs/{name}.json", data, immutable=True)
    folder = output / "inputs"
    folder.mkdir(parents=True, exist_ok=True)
    for role in ("train", "validation"):
        path = folder / f"{role}.jsonl"
        if path.exists() and digest(path) != source["source_hashes"][role + "_inputs"]:
            raise ValueError("Copied source input changed")
        if not path.exists(): copy2(source["stage"] / f"inputs/{role}.jsonl", path)
    result = {}
    for key, ref in source["models"].items():
        target = output / "models" / f"{key}.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) != ref["sha256"]: raise ValueError("Copied checkpoint changed")
        if not target.exists(): copy2(ref["path"], target)
        result[key] = dict(ref, path=target.relative_to(output).as_posix())
    write(output / "models.json", result, immutable=True)


def source_diagnostic(output, source):
    """Already observed training data only; no new PSO and no test geometry."""
    output = Path(output)
    missing = []
    for row in source["train"]:
        if row["has_label"]: continue
        job_id = row["case_id"].rsplit("_c", 1)[0]
        path = source["corpus"] / "labels" / job_id / "search" / (job_id + ".json")
        doc = read(path)
        case = next(c for c in doc["cases"] if c["case_id"] == row["case_id"])
        candidates = []
        for pool in case["pools"]:
            item = next(c for c in pool["candidates"] if c["candidate_id"] == pool["raw_candidate_id"])
            candidates.append({key: item[key] for key in ("candidate_id", "allocation", "search_rate", "search_dep_gamma", "search_feasible", "seed")})
        missing.append({key: row[key] for key in ("case_id", "layout_id", "n_public", "power_dbm", "xi")}
                       | {"search_dep_target": case["search_dep_target"], "best_by_seed": candidates,
                          "warning": "No returned label is not a proof of original-problem infeasibility; tightened search targets differ"})
    result = {"training_layout_cv_from_K2_input_bank": source["train_cv"], "low_cv_threshold": .9,
              "low_cv_layout_ids": source["low_cv_ids"], "missing_training_labels": missing,
              "new_pso_evaluations": 0, "reserved_test_generated": False,
              "scope": "Post-hoc development diagnostic, not final test or a proof excluding overfitting"}
    write(output / "source_diagnostic.json", result, immutable=True)
    return result
